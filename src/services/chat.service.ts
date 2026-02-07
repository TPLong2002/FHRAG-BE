import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { osClient } from "../lib/opensearch.js";
import { createEmbeddings } from "../lib/embeddings.js";
import { createLLM } from "../lib/llm.js";
import { OpenSearchHybridRetriever } from "../lib/hybrid-retriever.js";
import { config } from "../config/index.js";
import type { ChatRequest, ChatSource, EmbeddingProvider } from "../types/index.js";

const SYSTEM_PROMPT = `You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain relevant information, say so honestly.
Always cite which document the information comes from when possible.

Context:
{context}`;

const prompt = ChatPromptTemplate.fromMessages([
  ["system", SYSTEM_PROMPT],
  ["human", "{question}"],
]);

export async function chatWithSources(req: ChatRequest): Promise<{
  answer: string;
  sources: ChatSource[];
}> {
  const embeddings = createEmbeddings(
    config.embedding.defaultProvider as EmbeddingProvider,
    config.embedding.defaultModel
  );
  const llm = createLLM(req.provider, req.model);

  const retriever = new OpenSearchHybridRetriever({
    client: osClient,
    embeddings,
    indexName: config.opensearch.index,
    k: 4,
    pipeline: config.opensearch.pipelineName,
    documentIds: req.documentIds,
    userId: req.userId,
  });

  const docs = await retriever.invoke(req.question);

  const context = docs
    .map((d, i) => `[${i + 1}] (${d.metadata.fileName}) ${d.pageContent}`)
    .join("\n\n");

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
  onChunk: (text: string) => void
): Promise<ChatSource[]> {
  const embeddings = createEmbeddings(
    config.embedding.defaultProvider as EmbeddingProvider,
    config.embedding.defaultModel
  );
  const llm = createLLM(req.provider, req.model);

  const retriever = new OpenSearchHybridRetriever({
    client: osClient,
    embeddings,
    indexName: config.opensearch.index,
    k: 10,
    pipeline: config.opensearch.pipelineName,
    documentIds: req.documentIds,
    userId: req.userId,
  });

  const docs = await retriever.invoke(req.question);

  const context = docs
    .map((d, i) => `[${i + 1}] (${d.metadata.fileName}) ${d.pageContent}`)
    .join("\n\n");

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
