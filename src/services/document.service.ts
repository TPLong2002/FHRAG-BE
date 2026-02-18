import { v4 as uuidv4 } from "uuid";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createEmbeddings, getEmbeddingDimension } from "../lib/embeddings.js";
import { runQuery, initNeo4j } from "../lib/neo4j.js";
import { config } from "../config/index.js";
import { parseFile } from "./file-parser.service.js";
import {
  computeCrossDocumentSimilarity,
  computeDocumentRelationships,
} from "./graph.service.js";
import type { DocumentMeta, UploadOptions, AccessControl } from "../types/index.js";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: config.chunking.chunkSize,
  chunkOverlap: config.chunking.chunkOverlap,
});

export async function uploadDocument(
  filePath: string,
  originalName: string,
  mimeType: string,
  fileSize: number,
  options: UploadOptions,
): Promise<DocumentMeta> {
  const { embeddingProvider, embeddingModel, ownerId = "system" } = options;

  // Ensure Neo4j indexes exist with correct dimensions
  const dimension = getEmbeddingDimension(embeddingModel);
  await initNeo4j(dimension);

  // Parse file -> Document[] with page metadata
  const { docs: parsedDocs, fileType } = await parseFile(filePath, mimeType);
  if (!parsedDocs.length) throw new Error("No text content extracted from file");

  // Split into chunks (preserves original metadata like page number)
  const splitDocs = await splitter.splitDocuments(parsedDocs);
  if (!splitDocs.length) throw new Error("No chunks after splitting");

  const chunks = splitDocs.map((d) => d.pageContent);

  // Embed all chunks
  const embeddings = createEmbeddings(embeddingProvider, embeddingModel);
  const vectors = await embeddings.embedDocuments(chunks);

  // Create document metadata
  const documentId = uuidv4();
  const accessControl: AccessControl = {
    public: true,
    allowedUsers: [],
    allowedGroups: [],
  };

  const meta: DocumentMeta = {
    id: documentId,
    fileName: originalName,
    fileType,
    fileSize,
    totalChunks: chunks.length,
    ownerId,
    accessControl,
    embeddingProvider,
    embeddingModel,
    uploadedAt: new Date().toISOString(),
  };

  // Create Document node
  await runQuery(
    `CREATE (d:Document {
      documentId: $documentId, fileName: $fileName, fileType: $fileType,
      fileSize: $fileSize, totalChunks: $totalChunks, ownerId: $ownerId,
      accessControl: $accessControl,
      embeddingProvider: $embeddingProvider, embeddingModel: $embeddingModel,
      uploadedAt: $uploadedAt
    })`,
    {
      documentId,
      fileName: originalName,
      fileType,
      fileSize,
      totalChunks: chunks.length,
      ownerId,
      accessControl: JSON.stringify(accessControl),
      embeddingProvider,
      embeddingModel,
      uploadedAt: meta.uploadedAt,
    },
  );

  // Create Chunk nodes with embeddings + HAS_CHUNK relationships
  const chunkData = chunks.map((text, i) => ({
    chunkId: `${documentId}_chunk_${i}`,
    chunkIndex: i,
    text,
    embedding: vectors[i],
    fileName: originalName,
    fileType,
    ownerId,
    accessControlPublic: accessControl.public,
    allowedUsers: accessControl.allowedUsers,
  }));

  // Batch create in groups to avoid huge transactions
  const batchSize = 50;
  for (let start = 0; start < chunkData.length; start += batchSize) {
    const batch = chunkData.slice(start, start + batchSize);
    await runQuery(
      `MATCH (d:Document {documentId: $documentId})
       UNWIND $chunks AS chunk
       CREATE (c:Chunk {
         chunkId: chunk.chunkId, documentId: $documentId,
         chunkIndex: chunk.chunkIndex, text: chunk.text,
         embedding: chunk.embedding,
         fileName: chunk.fileName, fileType: chunk.fileType,
         ownerId: chunk.ownerId,
         accessControlPublic: chunk.accessControlPublic,
         allowedUsers: chunk.allowedUsers
       })
       CREATE (d)-[:HAS_CHUNK {position: chunk.chunkIndex}]->(c)`,
      { documentId, chunks: batch },
    );
  }

  // Create NEXT_CHUNK sequential links
  if (chunks.length > 1) {
    await runQuery(
      `MATCH (d:Document {documentId: $documentId})-[:HAS_CHUNK]->(c:Chunk)
       WITH c ORDER BY c.chunkIndex
       WITH collect(c) AS chunks
       UNWIND range(0, size(chunks) - 2) AS i
       WITH chunks[i] AS current, chunks[i + 1] AS next
       CREATE (current)-[:NEXT_CHUNK]->(next)`,
      { documentId },
    );
  }

  // Compute cross-document similarity + relationships (non-blocking)
  try {
    await computeCrossDocumentSimilarity(documentId, chunks, vectors);
    await computeDocumentRelationships(documentId);
  } catch (err) {
    console.error("Graph similarity error (non-blocking):", err);
  }

  return meta;
}

export async function listDocuments(userId?: string): Promise<DocumentMeta[]> {
  let cypher: string;
  const params: Record<string, unknown> = {};

  const returnFields = `RETURN d.documentId AS documentId, d.fileName AS fileName,
    d.fileType AS fileType, d.fileSize AS fileSize, d.totalChunks AS totalChunks,
    d.ownerId AS ownerId, d.accessControl AS accessControl,
    d.embeddingProvider AS embeddingProvider, d.embeddingModel AS embeddingModel,
    d.uploadedAt AS uploadedAt`;

  if (userId) {
    cypher = `MATCH (d:Document)
      WHERE d.ownerId = $userId
        OR d.accessControl CONTAINS '"public":true'
      ${returnFields} ORDER BY d.uploadedAt DESC LIMIT 1000`;
    params.userId = userId;
  } else {
    cypher = `MATCH (d:Document) ${returnFields} ORDER BY d.uploadedAt DESC LIMIT 1000`;
  }

  const results = await runQuery<{
    documentId: string;
    fileName: string;
    fileType: string;
    fileSize: number;
    totalChunks: number;
    ownerId: string;
    accessControl: string;
    embeddingProvider: string;
    embeddingModel: string;
    uploadedAt: string;
  }>(cypher, params);

  return results.map((d) => {
    let ac: AccessControl;
    try {
      ac = JSON.parse(d.accessControl);
    } catch {
      ac = { public: true, allowedUsers: [], allowedGroups: [] };
    }
    return {
      id: d.documentId,
      fileName: d.fileName,
      fileType: d.fileType,
      fileSize: d.fileSize,
      totalChunks: d.totalChunks,
      ownerId: d.ownerId,
      accessControl: ac,
      embeddingProvider: d.embeddingProvider,
      embeddingModel: d.embeddingModel,
      uploadedAt: d.uploadedAt,
    } as DocumentMeta;
  });
}

export async function deleteDocument(documentId: string): Promise<void> {
  await runQuery(
    `MATCH (d:Document {documentId: $documentId})
     OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
     DETACH DELETE c, d`,
    { documentId },
  );
}

export async function getDocument(documentId: string): Promise<DocumentMeta | null> {
  const results = await runQuery<{
    documentId: string;
    fileName: string;
    fileType: string;
    fileSize: number;
    totalChunks: number;
    ownerId: string;
    accessControl: string;
    embeddingProvider: string;
    embeddingModel: string;
    uploadedAt: string;
  }>(
    `MATCH (d:Document {documentId: $documentId})
     RETURN d.documentId AS documentId, d.fileName AS fileName,
            d.fileType AS fileType, d.fileSize AS fileSize, d.totalChunks AS totalChunks,
            d.ownerId AS ownerId, d.accessControl AS accessControl,
            d.embeddingProvider AS embeddingProvider, d.embeddingModel AS embeddingModel,
            d.uploadedAt AS uploadedAt`,
    { documentId },
  );

  if (!results.length) return null;

  const d = results[0];
  let ac: AccessControl;
  try {
    ac = JSON.parse(d.accessControl);
  } catch {
    ac = { public: true, allowedUsers: [], allowedGroups: [] };
  }

  return {
    id: d.documentId,
    fileName: d.fileName,
    fileType: d.fileType,
    fileSize: d.fileSize,
    totalChunks: d.totalChunks,
    ownerId: d.ownerId,
    accessControl: ac,
    embeddingProvider: d.embeddingProvider,
    embeddingModel: d.embeddingModel,
    uploadedAt: d.uploadedAt,
  } as DocumentMeta;
}
