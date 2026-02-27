import { OpenAIEmbeddings } from "@langchain/openai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import type { Embeddings } from "@langchain/core/embeddings";
import type { EmbeddingProvider } from "../types/index.js";
import { config } from "../config/index.js";

export function createEmbeddings(provider: EmbeddingProvider, model: string): Embeddings {
  switch (provider) {
    case "openai":
      return new OpenAIEmbeddings({
        apiKey: config.apiKeys.openai,
        model,
      });
    case "google":
      return new GoogleGenerativeAIEmbeddings({
        apiKey: config.apiKeys.google,
        model,
      });
    default:
      throw new Error(`Unsupported embedding provider: ${provider}`);
  }
}

export function getEmbeddingDimension(model: string): number {
  return config.embeddingDimensions[model] ?? 1536;
}

/** Available embedding models per provider */
export const EMBEDDING_MODELS: Record<EmbeddingProvider, { id: string; name: string }[]> = {
  openai: [
    { id: "text-embedding-3-large", name: "Text Embedding 3 Large (3072d)" },
    { id: "text-embedding-3-small", name: "Text Embedding 3 Small (1536d)" },
    { id: "text-embedding-ada-002", name: "Ada 002 (1536d)" },
  ],
  google: [
    // { id: "text-embedding-004", name: "Text Embedding 004 (768d)" },
    // { id: "embedding-001", name: "Embedding 001 (768d)" },
    { id: "gemini-embedding-001", name: "Gemini Embedding 001 (3072d)" },
  ],
};
