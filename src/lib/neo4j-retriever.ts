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
    super();
    this.embeddings = fields.embeddings;
    this.k = fields.k ?? config.search.topK;
    this.documentIds = fields.documentIds;
    this.userId = fields.userId;
  }

  async _getRelevantDocuments(query: string): Promise<Document[]> {
    const queryVector = await this.embeddings.embedQuery(query);
    // Fetch more candidates than final k to account for dedup + filtering
    const candidateK = neo4j.int(this.k * 3);

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
      { topK: candidateK, vector: queryVector },
    );

    // 2. Full-text search (sanitize query for Lucene)
    const sanitizedQuery = query.replace(/[+\-&|!(){}[\]^"~*?:\\/]/g, " ").trim();
    const fulltextResults = sanitizedQuery
      ? await runQuery<{
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
          { query: sanitizedQuery, topK: candidateK },
        )
      : [];

    // 3. Build lookup for original vector cosine scores
    const vectorScoreMap = new Map<string, number>();
    for (const r of vectorResults) {
      vectorScoreMap.set(r.chunkId, r.score);
    }

    // 4. Merge with RRF (Reciprocal Rank Fusion) for ranking
    const rrf = new Map<string, { rrfScore: number; data: (typeof vectorResults)[0] }>();
    const rk = 60; // RRF constant

    vectorResults.forEach((r, i) => {
      const rrfScore = config.search.vectorWeight / (rk + i + 1);
      rrf.set(r.chunkId, { rrfScore, data: r });
    });

    fulltextResults.forEach((r, i) => {
      const rrfScore = config.search.fulltextWeight / (rk + i + 1);
      const existing = rrf.get(r.chunkId);
      if (existing) {
        existing.rrfScore += rrfScore;
      } else {
        rrf.set(r.chunkId, { rrfScore, data: r });
      }
    });

    // 5. Sort by RRF score (for ranking only)
    let results = [...rrf.values()].sort((a, b) => b.rrfScore - a.rrfScore);

    // 6. Permission filter
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

    // 7. Document ID filter
    if (this.documentIds?.length) {
      results = results.filter((r) => this.documentIds!.includes(r.data.documentId));
    }

    // 8. Take top-k and convert to LangChain Documents
    //    Return original cosine similarity as _score (0-1 range) for display
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
          _score: vectorScoreMap.get(d.chunkId) ?? d.score,
        },
      });
    });
  }
}
