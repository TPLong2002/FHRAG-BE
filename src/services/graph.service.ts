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

/**
 * Get table schema context for chunks (via MENTIONS_TABLE relationships).
 * Returns table definitions + FK relationships as structured text for LLM context.
 */
export async function getTableContextForChunks(chunkIds: string[]): Promise<string> {
  if (!chunkIds.length) return "";

  // Find tables mentioned by these chunks + their FK-connected tables
  const tables = await runQuery<{
    name: string;
    displayName: string;
    description: string;
    columns: string;
  }>(
    `UNWIND $chunkIds AS cid
     MATCH (c:Chunk {chunkId: cid})-[:MENTIONS_TABLE]->(t:Table)
     RETURN DISTINCT t.name AS name, t.displayName AS displayName,
            t.description AS description, t.columns AS columns`,
    { chunkIds },
  );

  if (!tables.length) return "";

  const tableNames = tables.map((t) => t.name);

  // Also fetch FK-connected tables not yet in the list
  const fkTables = await runQuery<{
    name: string;
    displayName: string;
    description: string;
    columns: string;
    fkFrom: string;
    fkFromCol: string;
    fkToCol: string;
    direction: string;
  }>(
    `UNWIND $names AS tName
     MATCH (t:Table {name: tName})-[fk:FOREIGN_KEY]->(other:Table)
     WHERE NOT other.name IN $names
     RETURN DISTINCT other.name AS name, other.displayName AS displayName,
            other.description AS description, other.columns AS columns,
            t.name AS fkFrom, fk.fromColumn AS fkFromCol, fk.toColumn AS fkToCol, 'out' AS direction
     UNION
     UNWIND $names AS tName
     MATCH (other:Table)-[fk:FOREIGN_KEY]->(t:Table {name: tName})
     WHERE NOT other.name IN $names
     RETURN DISTINCT other.name AS name, other.displayName AS displayName,
            other.description AS description, other.columns AS columns,
            t.name AS fkFrom, fk.fromColumn AS fkFromCol, fk.toColumn AS fkToCol, 'in' AS direction`,
    { names: tableNames },
  );

  // Get FK relationships between known tables
  const allNames = [...new Set([...tableNames, ...fkTables.map((t) => t.name)])];
  const fks = await runQuery<{
    from: string;
    to: string;
    fromCol: string;
    toCol: string;
  }>(
    `MATCH (t1:Table)-[fk:FOREIGN_KEY]->(t2:Table)
     WHERE t1.name IN $names AND t2.name IN $names
     RETURN t1.name AS from, t2.name AS to,
            fk.fromColumn AS fromCol, fk.toColumn AS toCol`,
    { names: allNames },
  );

  // Format as structured text
  const allTables = [...tables, ...fkTables];
  const seen = new Set<string>();
  const lines: string[] = ["=== DATABASE SCHEMA CONTEXT ==="];

  for (const t of allTables) {
    if (seen.has(t.name)) continue;
    seen.add(t.name);

    lines.push(`\nTable: ${t.displayName}`);
    if (t.description) lines.push(`  Description: ${t.description}`);

    try {
      const cols = JSON.parse(t.columns) as Array<{
        name: string; type: string; nullable: boolean; isPrimaryKey: boolean; description?: string;
      }>;
      lines.push("  Columns:");
      for (const col of cols) {
        const pk = col.isPrimaryKey ? " [PK]" : "";
        const nullable = col.nullable ? " NULL" : " NOT NULL";
        const desc = col.description ? ` -- ${col.description}` : "";
        lines.push(`    - ${col.name} ${col.type}${pk}${nullable}${desc}`);
      }
    } catch { /* skip */ }
  }

  if (fks.length > 0) {
    lines.push("\nForeign Key Relationships:");
    for (const fk of fks) {
      lines.push(`  ${fk.from}.${fk.fromCol} -> ${fk.to}.${fk.toCol}`);
    }
  }

  return lines.join("\n");
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

/**
 * Get schema-level graph: Table nodes + FOREIGN_KEY edges.
 */
export async function getSchemaGraph(documentId?: string): Promise<GraphData> {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeSet = new Set<string>();

  if (documentId) {
    const results = await runQuery<{
      tableName: string;
      displayName: string;
      description: string;
      columns: string;
      relatedTable: string | null;
      relatedDisplayName: string | null;
      relatedDescription: string | null;
      relatedColumns: string | null;
      fkFromCol: string | null;
      fkToCol: string | null;
      fkDirection: string | null;
    }>(
      `MATCH (d:Document {documentId: $documentId})-[:HAS_TABLE]->(t:Table)
       OPTIONAL MATCH (t)-[fk:FOREIGN_KEY]->(other:Table)
       RETURN t.name AS tableName, t.displayName AS displayName,
              t.description AS description, t.columns AS columns,
              other.name AS relatedTable, other.displayName AS relatedDisplayName,
              other.description AS relatedDescription, other.columns AS relatedColumns,
              fk.fromColumn AS fkFromCol, fk.toColumn AS fkToCol, 'out' AS fkDirection
       UNION
       MATCH (d:Document {documentId: $documentId})-[:HAS_TABLE]->(t:Table)
       OPTIONAL MATCH (other:Table)-[fk:FOREIGN_KEY]->(t)
       WHERE other IS NOT NULL
       RETURN t.name AS tableName, t.displayName AS displayName,
              t.description AS description, t.columns AS columns,
              other.name AS relatedTable, other.displayName AS relatedDisplayName,
              other.description AS relatedDescription, other.columns AS relatedColumns,
              fk.fromColumn AS fkFromCol, fk.toColumn AS fkToCol, 'in' AS fkDirection`,
      { documentId },
    );

    for (const row of results) {
      if (!nodeSet.has(row.tableName)) {
        nodeSet.add(row.tableName);
        nodes.push({
          id: row.tableName,
          label: row.displayName,
          type: "table",
          properties: { description: row.description, columns: row.columns },
        });
      }
      if (row.relatedTable && !nodeSet.has(row.relatedTable)) {
        nodeSet.add(row.relatedTable);
        nodes.push({
          id: row.relatedTable,
          label: row.relatedDisplayName!,
          type: "table",
          properties: { description: row.relatedDescription, columns: row.relatedColumns },
        });
      }
      if (row.relatedTable && row.fkFromCol) {
        const source = row.fkDirection === "out" ? row.tableName : row.relatedTable;
        const target = row.fkDirection === "out" ? row.relatedTable : row.tableName;
        edges.push({
          source,
          target,
          type: "FOREIGN_KEY",
          properties: { fromColumn: row.fkFromCol, toColumn: row.fkToCol },
        });
      }
    }
  } else {
    const tables = await runQuery<{
      name: string;
      displayName: string;
      description: string;
      columns: string;
    }>(
      `MATCH (t:Table)
       RETURN t.name AS name, t.displayName AS displayName,
              t.description AS description, t.columns AS columns
       LIMIT 200`,
    );

    for (const t of tables) {
      nodeSet.add(t.name);
      nodes.push({
        id: t.name,
        label: t.displayName,
        type: "table",
        properties: { description: t.description, columns: t.columns },
      });
    }

    const fks = await runQuery<{
      from: string;
      to: string;
      fromCol: string;
      toCol: string;
    }>(
      `MATCH (t1:Table)-[fk:FOREIGN_KEY]->(t2:Table)
       WHERE t1.name IN $names AND t2.name IN $names
       RETURN t1.name AS from, t2.name AS to,
              fk.fromColumn AS fromCol, fk.toColumn AS toCol`,
      { names: [...nodeSet] },
    );

    for (const fk of fks) {
      edges.push({
        source: fk.from,
        target: fk.to,
        type: "FOREIGN_KEY",
        properties: { fromColumn: fk.fromCol, toColumn: fk.toCol },
      });
    }
  }

  // Deduplicate edges
  const edgeSet = new Set<string>();
  const uniqueEdges = edges.filter((e) => {
    const key = `${e.source}->${e.target}:${e.properties.fromColumn}->${e.properties.toColumn}`;
    if (edgeSet.has(key)) return false;
    edgeSet.add(key);
    return true;
  });

  return { nodes, edges: uniqueEdges };
}

export async function deleteTable(tableName: string): Promise<void> {
  await runQuery(
    `MATCH (t:Table {name: $name}) DETACH DELETE t`,
    { name: tableName },
  );
}

export async function deleteForeignKey(
  fromTable: string,
  toTable: string,
  fromColumn: string,
  toColumn: string,
): Promise<void> {
  await runQuery(
    `MATCH (from:Table {name: $fromTable})-[fk:FOREIGN_KEY {fromColumn: $fromColumn, toColumn: $toColumn}]->(to:Table {name: $toTable})
     DELETE fk`,
    { fromTable, toTable, fromColumn, toColumn },
  );
}
