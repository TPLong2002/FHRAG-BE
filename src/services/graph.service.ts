import neo4j from "neo4j-driver";
import { runQuery } from "../lib/neo4j.js";
import { config } from "../config/index.js";
import type {
  ChunkNeighbors,
  SimilarityPair,
  RelatedDocument,
  GraphData,
  GraphNode,
  GraphEdge,
} from "../types/index.js";

// ==================== WRITE OPERATIONS ====================

/**
 * Compute cross-document SIMILAR_TO edges using Neo4j vector index.
 */
export async function computeCrossDocumentSimilarity(
  documentId: string,
  chunkTexts: string[],
  vectors: number[][],
): Promise<void> {
  const topK = config.graph.similarityTopK;
  const threshold = config.graph.similarityThreshold;
  const pairs: SimilarityPair[] = [];

  for (let i = 0; i < chunkTexts.length; i++) {
    const sourceChunkId = `${documentId}_chunk_${i}`;

    try {
      const results = await runQuery<{
        chunkId: string;
        docId: string;
        chunkIndex: number;
        score: number;
      }>(
        `CALL db.index.vector.queryNodes('chunk_embeddings', $topK, $vector)
         YIELD node, score
         WHERE node.documentId <> $documentId
         RETURN node.chunkId AS chunkId, node.documentId AS docId,
                node.chunkIndex AS chunkIndex, score
         LIMIT $topK`,
        { topK: neo4j.int(topK), vector: vectors[i], documentId },
      );

      for (const r of results.slice(0, topK)) {
        if (r.score < threshold) continue;
        pairs.push({
          sourceChunkId,
          targetChunkId: r.chunkId,
          score: r.score,
        });
      }
    } catch (err) {
      console.error(`Similarity search failed for chunk ${i}:`, err);
    }
  }

  if (pairs.length > 0) {
    await runQuery(
      `UNWIND $pairs AS pair
       MATCH (a:Chunk {chunkId: pair.sourceChunkId})
       MATCH (b:Chunk {chunkId: pair.targetChunkId})
       MERGE (a)-[r:SIMILAR_TO]->(b)
       SET r.score = pair.score`,
      { pairs },
    );
  }
}

/**
 * Aggregate SIMILAR_TO edges into document-level RELATED_TO.
 */
export async function computeDocumentRelationships(documentId: string): Promise<void> {
  const minConnections = config.graph.minRelatedConnections;

  await runQuery(
    `MATCH (d1:Document {documentId: $documentId})-[:HAS_CHUNK]->(:Chunk)-[s:SIMILAR_TO]-(:Chunk)<-[:HAS_CHUNK]-(d2:Document)
     WHERE d1 <> d2
     WITH d1, d2, count(s) AS connectionCount, avg(s.score) AS avgScore
     WHERE connectionCount >= $minConnections
     MERGE (d1)-[r:RELATED_TO]->(d2)
     SET r.score = avgScore, r.connectionCount = connectionCount`,
    { documentId, minConnections: neo4j.int(minConnections) },
  );
}

// ==================== READ OPERATIONS (Retrieval Enhancement) ====================

/**
 * Fetch prev/next chunks for given chunk IDs.
 */
export async function getNeighborChunks(chunkIds: string[]): Promise<ChunkNeighbors[]> {
  return runQuery<ChunkNeighbors>(
    `UNWIND $chunkIds AS cid
     MATCH (c:Chunk {chunkId: cid})
     OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
     OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
     RETURN c.chunkId AS chunkId,
            prev.chunkId AS prevChunkId, prev.text AS prevText, prev.chunkIndex AS prevIndex,
            next.chunkId AS nextChunkId, next.text AS nextText, next.chunkIndex AS nextIndex`,
    { chunkIds },
  );
}

/**
 * Fetch cross-document similar chunks.
 */
export async function getSimilarChunksFromGraph(
  chunkIds: string[],
  limit: number = 3,
): Promise<
  Array<{
    chunkId: string;
    text: string;
    documentId: string;
    fileName: string;
    chunkIndex: number;
    similarityScore: number;
  }>
> {
  return runQuery(
    `UNWIND $chunkIds AS cid
     MATCH (c:Chunk {chunkId: cid})-[s:SIMILAR_TO]-(related:Chunk)
     WHERE NOT related.chunkId IN $chunkIds
     RETURN DISTINCT related.chunkId AS chunkId, related.text AS text,
            related.documentId AS documentId, related.fileName AS fileName,
            related.chunkIndex AS chunkIndex, s.score AS similarityScore
     ORDER BY s.score DESC
     LIMIT $limit`,
    { chunkIds, limit: neo4j.int(limit) },
  );
}

// ==================== READ OPERATIONS (API/Visualization) ====================

/**
 * Get documents related to a specific document.
 */
export async function getRelatedDocuments(documentId: string): Promise<RelatedDocument[]> {
  return runQuery<RelatedDocument>(
    `MATCH (d:Document {documentId: $documentId})-[r:RELATED_TO]-(other:Document)
     RETURN other.documentId AS documentId, other.fileName AS fileName,
            r.score AS score, r.connectionCount AS connectionCount
     ORDER BY r.score DESC`,
    { documentId },
  );
}

/**
 * Get document-level graph for visualization.
 */
export async function getDocumentGraph(documentId?: string): Promise<GraphData> {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeSet = new Set<string>();

  if (documentId) {
    const results = await runQuery<{
      docId: string;
      docName: string;
      docType: string;
      chunks: number;
      relDocId: string | null;
      relDocName: string | null;
      relScore: number | null;
      relCount: number | null;
    }>(
      `MATCH (d:Document {documentId: $documentId})
       OPTIONAL MATCH (d)-[r:RELATED_TO]-(other:Document)
       RETURN d.documentId AS docId, d.fileName AS docName, d.fileType AS docType,
              d.totalChunks AS chunks,
              other.documentId AS relDocId, other.fileName AS relDocName,
              r.score AS relScore, r.connectionCount AS relCount`,
      { documentId },
    );

    for (const row of results) {
      if (!nodeSet.has(row.docId)) {
        nodeSet.add(row.docId);
        nodes.push({
          id: row.docId,
          label: row.docName,
          type: "document",
          properties: { fileType: row.docType, totalChunks: row.chunks },
        });
      }
      if (row.relDocId && !nodeSet.has(row.relDocId)) {
        nodeSet.add(row.relDocId);
        nodes.push({
          id: row.relDocId,
          label: row.relDocName!,
          type: "document",
          properties: {},
        });
      }
      if (row.relDocId) {
        edges.push({
          source: row.docId,
          target: row.relDocId,
          type: "RELATED_TO",
          properties: { score: row.relScore, connectionCount: row.relCount },
        });
      }
    }
  } else {
    const results = await runQuery<{
      docId: string;
      docName: string;
      docType: string;
      chunks: number;
    }>(
      `MATCH (d:Document)
       RETURN d.documentId AS docId, d.fileName AS docName, d.fileType AS docType,
              d.totalChunks AS chunks
       ORDER BY d.uploadedAt DESC LIMIT 50`,
    );

    for (const row of results) {
      nodes.push({
        id: row.docId,
        label: row.docName,
        type: "document",
        properties: { fileType: row.docType, totalChunks: row.chunks },
      });
      nodeSet.add(row.docId);
    }

    const rels = await runQuery<{
      source: string;
      target: string;
      score: number;
      count: number;
    }>(
      `MATCH (d1:Document)-[r:RELATED_TO]->(d2:Document)
       WHERE d1.documentId IN $ids AND d2.documentId IN $ids
       RETURN d1.documentId AS source, d2.documentId AS target,
              r.score AS score, r.connectionCount AS count`,
      { ids: [...nodeSet] },
    );

    for (const rel of rels) {
      edges.push({
        source: rel.source,
        target: rel.target,
        type: "RELATED_TO",
        properties: { score: rel.score, connectionCount: rel.count },
      });
    }
  }

  return { nodes, edges };
}

/**
 * Get chunk-level graph for a document.
 */
export async function getChunkGraph(documentId: string): Promise<GraphData> {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeSet = new Set<string>();

  const chunks = await runQuery<{
    chunkId: string;
    chunkIndex: number;
    fileName: string;
    textPreview: string;
  }>(
    `MATCH (d:Document {documentId: $documentId})-[:HAS_CHUNK]->(c:Chunk)
     RETURN c.chunkId AS chunkId, c.chunkIndex AS chunkIndex,
            c.fileName AS fileName, left(c.text, 100) AS textPreview
     ORDER BY c.chunkIndex`,
    { documentId },
  );

  for (const c of chunks) {
    nodeSet.add(c.chunkId);
    nodes.push({
      id: c.chunkId,
      label: `Chunk ${c.chunkIndex}`,
      type: "chunk",
      properties: { fileName: c.fileName, textPreview: c.textPreview },
    });
  }

  const nextLinks = await runQuery<{ from: string; to: string }>(
    `MATCH (d:Document {documentId: $documentId})-[:HAS_CHUNK]->(c:Chunk)-[:NEXT_CHUNK]->(n:Chunk)
     RETURN c.chunkId AS from, n.chunkId AS to`,
    { documentId },
  );

  for (const link of nextLinks) {
    edges.push({ source: link.from, target: link.to, type: "NEXT_CHUNK", properties: {} });
  }

  const simLinks = await runQuery<{
    from: string;
    to: string;
    toFileName: string;
    score: number;
  }>(
    `MATCH (d:Document {documentId: $documentId})-[:HAS_CHUNK]->(c:Chunk)-[s:SIMILAR_TO]-(other:Chunk)
     WHERE other.documentId <> $documentId
     RETURN c.chunkId AS from, other.chunkId AS to, other.fileName AS toFileName, s.score AS score
     LIMIT 20`,
    { documentId },
  );

  for (const link of simLinks) {
    if (!nodeSet.has(link.to)) {
      nodeSet.add(link.to);
      nodes.push({
        id: link.to,
        label: `${link.toFileName} (external)`,
        type: "chunk",
        properties: { external: true },
      });
    }
    edges.push({
      source: link.from,
      target: link.to,
      type: "SIMILAR_TO",
      properties: { score: link.score },
    });
  }

  return { nodes, edges };
}
