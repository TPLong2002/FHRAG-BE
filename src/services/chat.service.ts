import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { createEmbeddings } from "../lib/embeddings.js";
import { createLLM } from "../lib/llm.js";
import { Neo4jHybridRetriever } from "../lib/neo4j-retriever.js";
import { getNeighborChunks, getSimilarChunksFromGraph } from "./graph.service.js";
import { config } from "../config/index.js";
import type { ChatRequest, ChatSource, EmbeddingProvider } from "../types/index.js";

const SYSTEM_PROMPT = `You are a knowledgeable assistant specialized in answering questions based on provided documents.

Instructions:
- Answer ONLY based on the provided context. Do not make up information.
- If the context doesn't contain enough information, say so clearly.
- When information comes from multiple documents, synthesize the answer and cite each source.
- Reference sources by their document name, e.g. "(from filename.pdf)".
- For technical/database questions, be precise with table names, column names, relationships, and data types.
- If context chunks are labeled [neighbor] or [similar], they provide additional related context.

Context:
{context}`;

const prompt = ChatPromptTemplate.fromMessages([
  ["system", SYSTEM_PROMPT],
  ["human", "{question}"],
]);

/**
 * Build context string from docs, sorted by document then chunk index for coherence.
 */
function buildContext(docs: Document[]): string {
  const sorted = [...docs].sort((a, b) => {
    const docCmp = String(a.metadata.documentId).localeCompare(String(b.metadata.documentId));
    if (docCmp !== 0) return docCmp;
    return (a.metadata.chunkIndex as number) - (b.metadata.chunkIndex as number);
  });

  return sorted
    .map((d, i) => {
      const tag = d.metadata._graphSource ? ` [${d.metadata._graphSource}]` : "";
      return `[${i + 1}] (${d.metadata.fileName || "unknown"})${tag} ${d.pageContent}`;
    })
    .join("\n\n");
}

/**
 * Enhance retrieved docs with graph context (neighbors + cross-doc similar).
 */
async function enhanceWithGraphContext(docs: Document[]): Promise<Document[]> {
  try {
    const chunkIds = docs.map((d) => {
      const docId = d.metadata.documentId as string;
      const idx = d.metadata.chunkIndex as number;
      return `${docId}_chunk_${idx}`;
    });

    const seenChunkIds = new Set(chunkIds);
    const additional: Document[] = [];

    // 1. Neighbor chunks (prev/next)
    const neighbors = await getNeighborChunks(chunkIds);
    for (const n of neighbors) {
      if (n.prevChunkId && !seenChunkIds.has(n.prevChunkId) && n.prevText) {
        seenChunkIds.add(n.prevChunkId);
        const [docId] = n.prevChunkId.split("_chunk_");
        additional.push(
          new Document({
            pageContent: n.prevText,
            metadata: { documentId: docId, chunkIndex: n.prevIndex, _graphSource: "neighbor" },
          }),
        );
      }
      if (n.nextChunkId && !seenChunkIds.has(n.nextChunkId) && n.nextText) {
        seenChunkIds.add(n.nextChunkId);
        const [docId] = n.nextChunkId.split("_chunk_");
        additional.push(
          new Document({
            pageContent: n.nextText,
            metadata: { documentId: docId, chunkIndex: n.nextIndex, _graphSource: "neighbor" },
          }),
        );
      }
    }

    // 2. Cross-document similar chunks
    const similar = await getSimilarChunksFromGraph(chunkIds, 3);
    for (const sc of similar) {
      if (!seenChunkIds.has(sc.chunkId)) {
        seenChunkIds.add(sc.chunkId);
        additional.push(
          new Document({
            pageContent: sc.text,
            metadata: {
              documentId: sc.documentId,
              fileName: sc.fileName,
              chunkIndex: sc.chunkIndex,
              _graphSource: "similar",
            },
          }),
        );
      }
    }

    return [...docs, ...additional];
  } catch (err) {
    console.error("Graph enhancement failed:", err);
    return docs;
  }
}

export async function chatWithSources(req: ChatRequest): Promise<{
  answer: string;
  sources: ChatSource[];
}> {
  const embeddings = createEmbeddings(
    config.embedding.defaultProvider as EmbeddingProvider,
    config.embedding.defaultModel,
  );
  const llm = createLLM(req.provider, req.model);

  const retriever = new Neo4jHybridRetriever({
    embeddings,
    k: config.search.topK,
    documentIds: req.documentIds,
    userId: req.userId,
  });

  const docs = await retriever.invoke(req.question);
  const enhancedDocs = await enhanceWithGraphContext(docs);

  const context = buildContext(enhancedDocs);

  const chain = prompt.pipe(llm).pipe(new StringOutputParser());
  const answer = await chain.invoke({ context, question: req.question });

  const sources: ChatSource[] = docs.map((d) => ({
    documentId: d.metadata.documentId as string,
    fileName: d.metadata.fileName as string,
    chunkIndex: d.metadata.chunkIndex as number,
    content: d.pageContent,
    score: d.metadata._score as number,
  }));

  return { answer, sources };
}

/** Streaming version - yields text chunks via callback */
export async function chatStream(
  req: ChatRequest,
  onChunk: (text: string) => void,
): Promise<ChatSource[]> {
  const embeddings = createEmbeddings(
    config.embedding.defaultProvider as EmbeddingProvider,
    config.embedding.defaultModel,
  );
  const llm = createLLM(req.provider, req.model);

  const retriever = new Neo4jHybridRetriever({
    embeddings,
    k: config.search.topK,
    documentIds: req.documentIds,
    userId: req.userId,
  });

  const docs = await retriever.invoke(req.question);
  const enhancedDocs = await enhanceWithGraphContext(docs);

  const context = buildContext(enhancedDocs);

  const chain = prompt.pipe(llm).pipe(new StringOutputParser());
  const stream = await chain.stream({ context, question: req.question });

  for await (const chunk of stream) {
    onChunk(chunk);
  }

  return docs.map((d) => ({
    documentId: d.metadata.documentId as string,
    fileName: d.metadata.fileName as string,
    chunkIndex: d.metadata.chunkIndex as number,
    content: d.pageContent,
    score: d.metadata._score as number,
  }));
}
