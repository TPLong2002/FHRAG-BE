import neo4j from "neo4j-driver";
import { BaseRetriever } from "@langchain/core/retrievers";
import { Document } from "@langchain/core/documents";
import type { Embeddings } from "@langchain/core/embeddings";
import { runQuery } from "./neo4j.js";
import { config } from "../config/index.js";

export interface Neo4jHybridRetrieverOptions {
  embeddings: Embeddings;
  k?: number;
  documentIds?: string[];
  userId?: string;
}

/**
 * Neo4j hybrid retriever combining vector search + full-text search.
 * Uses Reciprocal Rank Fusion (RRF) to merge results.
 */
export class Neo4jHybridRetriever extends BaseRetriever {
  lc_namespace = ["custom", "neo4j"];

  private embeddings: Embeddings;
  private k: number;
  private documentIds?: string[];
  private userId?: string;

  constructor(fields: Neo4jHybridRetrieverOptions) {
    super(fields);
    this.embeddings = fields.embeddings;
    this.k = fields.k ?? config.search.topK;
    this.documentIds = fields.documentIds;
    this.userId = fields.userId;
  }

  async _getRelevantDocuments(query: string): Promise<Document[]> {
    const queryVector = await this.embeddings.embedQuery(query);

    // 1. Vector search
    const vectorResults = await runQuery<{
      chunkId: string;
      text: string;
      documentId: string;
      fileName: string;
      fileType: string;
      chunkIndex: number;
      ownerId: string;
      accessControlPublic: boolean;
      allowedUsers: string[];
      score: number;
    }>(
      `CALL db.index.vector.queryNodes('chunk_embeddings', $topK, $vector)
       YIELD node, score
       RETURN node.chunkId AS chunkId, node.text AS text,
              node.documentId AS documentId, node.fileName AS fileName,
              node.fileType AS fileType, node.chunkIndex AS chunkIndex,
              node.ownerId AS ownerId,
              node.accessControlPublic AS accessControlPublic,
              node.allowedUsers AS allowedUsers,
              score`,
      { topK: neo4j.int(this.k), vector: queryVector },
    );

    // 2. Full-text search
    const fulltextResults = await runQuery<{
      chunkId: string;
      text: string;
      documentId: string;
      fileName: string;
      fileType: string;
      chunkIndex: number;
      ownerId: string;
      accessControlPublic: boolean;
      allowedUsers: string[];
      score: number;
    }>(
      `CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
       YIELD node, score
       WITH node, score LIMIT $topK
       RETURN node.chunkId AS chunkId, node.text AS text,
              node.documentId AS documentId, node.fileName AS fileName,
              node.fileType AS fileType, node.chunkIndex AS chunkIndex,
              node.ownerId AS ownerId,
              node.accessControlPublic AS accessControlPublic,
              node.allowedUsers AS allowedUsers,
              score`,
      { query, topK: neo4j.int(this.k) },
    );

    // 3. Merge with RRF (Reciprocal Rank Fusion)
    const rrf = new Map<string, { score: number; data: (typeof vectorResults)[0] }>();
    const rk = 60; // RRF constant

    vectorResults.forEach((r, i) => {
      const rrfScore = config.search.vectorWeight / (rk + i + 1);
      rrf.set(r.chunkId, { score: rrfScore, data: r });
    });

    fulltextResults.forEach((r, i) => {
      const rrfScore = config.search.fulltextWeight / (rk + i + 1);
      const existing = rrf.get(r.chunkId);
      if (existing) {
        existing.score += rrfScore;
      } else {
        rrf.set(r.chunkId, { score: rrfScore, data: r });
      }
    });

    // 4. Sort by RRF score
    let results = [...rrf.values()].sort((a, b) => b.score - a.score);

    // 5. Permission filter
    if (this.userId) {
      results = results.filter((r) => {
        const d = r.data;
        return (
          d.accessControlPublic === true ||
          d.ownerId === this.userId ||
          (d.allowedUsers && d.allowedUsers.includes(this.userId!))
        );
      });
    }

    // 6. Document ID filter
    if (this.documentIds?.length) {
      results = results.filter((r) => this.documentIds!.includes(r.data.documentId));
    }

    // 7. Take top-k and convert to LangChain Documents
    return results.slice(0, this.k).map((r) => {
      const d = r.data;
      return new Document({
        pageContent: d.text,
        metadata: {
          documentId: d.documentId,
          fileName: d.fileName,
          fileType: d.fileType,
          chunkIndex: d.chunkIndex,
          ownerId: d.ownerId,
          _score: r.score,
        },
      });
    });
  }
}
