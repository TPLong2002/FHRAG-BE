import "dotenv/config";

export const config = {
  port: parseInt(process.env.PORT || "3001"),

  opensearch: {
    url: process.env.OPENSEARCH_URL || "https://localhost:9200",
    username: process.env.OPENSEARCH_USERNAME || "admin",
    password: process.env.OPENSEARCH_PASSWORD || "StrongPassword123!",
    index: process.env.OPENSEARCH_INDEX || "rag_documents",
    metaIndex: process.env.OPENSEARCH_META_INDEX || "rag_documents_meta",
    pipelineName: "hybrid-search-pipeline",
  },

  apiKeys: {
    openai: process.env.OPENAI_API_KEY || "",
    google: process.env.GOOGLE_API_KEY || "",
  },

  embedding: {
    defaultProvider: process.env.DEFAULT_EMBEDDING_PROVIDER || "openai",
    defaultModel: process.env.DEFAULT_EMBEDDING_MODEL || "text-embedding-3-small",
  },

  chunking: {
    chunkSize: parseInt(process.env.CHUNK_SIZE || "1000"),
    chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || "200"),
  },

  embeddingDimensions: {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-004": 768,
    "embedding-001": 768,
  } as Record<string, number>,
} as const;
