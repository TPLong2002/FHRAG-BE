import { v4 as uuidv4 } from "uuid";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { osClient } from "../lib/opensearch.js";
import { createEmbeddings, getEmbeddingDimension } from "../lib/embeddings.js";
import { initOpenSearch } from "../lib/opensearch.js";
import { config } from "../config/index.js";
import { parseFile } from "./file-parser.service.js";
import type { DocumentMeta, UploadOptions, AccessControl, EmbeddingProvider } from "../types/index.js";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: config.chunking.chunkSize,
  chunkOverlap: config.chunking.chunkOverlap,
});

export async function uploadDocument(
  filePath: string,
  originalName: string,
  mimeType: string,
  fileSize: number,
  options: UploadOptions
): Promise<DocumentMeta> {
  const { embeddingProvider, embeddingModel, ownerId = "system" } = options;

  // Ensure index exists with correct dimensions
  const dimension = getEmbeddingDimension(embeddingModel);
  await initOpenSearch(dimension);

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

  // Store document metadata
  await osClient.index({
    index: config.opensearch.metaIndex,
    id: documentId,
    body: meta,
    refresh: "true",
  });

  // Store chunks with embeddings (bulk)
  const bulkBody: unknown[] = [];
  for (let i = 0; i < chunks.length; i++) {
    bulkBody.push({ index: { _index: config.opensearch.index } });
    bulkBody.push({
      text: chunks[i],
      embedding: vectors[i],
      metadata: {
        documentId,
        fileName: originalName,
        fileType,
        chunkIndex: i,
        ownerId,
        accessControl,
      },
    });
  }

  const bulkResult = await osClient.bulk({ body: bulkBody, refresh: "true" });
  if (bulkResult.body.errors) {
    console.error("Bulk indexing had errors:", JSON.stringify(bulkResult.body.items.slice(0, 3)));
  }

  return meta;
}

export async function listDocuments(userId?: string): Promise<DocumentMeta[]> {
  const query: Record<string, unknown> = userId
    ? {
        bool: {
          should: [
            { term: { "accessControl.public": true } },
            { term: { "accessControl.allowedUsers": userId } },
            { term: { ownerId: userId } },
          ],
          minimum_should_match: 1,
        },
      }
    : { match_all: {} };

  const result = await osClient.search({
    index: config.opensearch.metaIndex,
    body: { query, size: 1000, sort: [{ uploadedAt: "desc" }] },
  });

  return result.body.hits.hits.map((hit: Record<string, unknown>) => ({
    ...(hit._source as DocumentMeta),
    id: hit._id as string,
  }));
}

export async function deleteDocument(documentId: string): Promise<void> {
  // Delete all chunks belonging to this document
  await osClient.deleteByQuery({
    index: config.opensearch.index,
    body: { query: { term: { "metadata.documentId": documentId } } },
    refresh: true,
  });

  // Delete metadata
  await osClient.delete({
    index: config.opensearch.metaIndex,
    id: documentId,
    refresh: "true",
  });
}

export async function getDocument(documentId: string): Promise<DocumentMeta | null> {
  try {
    const result = await osClient.get({ index: config.opensearch.metaIndex, id: documentId });
    return { ...(result.body._source as DocumentMeta), id: result.body._id as string };
  } catch {
    return null;
  }
}
