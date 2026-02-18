import "dotenv/config";

export const config = {
  port: parseInt(process.env.PORT || "3001"),

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

  neo4j: {
    uri: process.env.NEO4J_URI || "bolt://localhost:7687",
    username: process.env.NEO4J_USERNAME || "neo4j",
    password: process.env.NEO4J_PASSWORD || "neo4j_password",
    database: process.env.NEO4J_DATABASE || "neo4j",
  },

  search: {
    topK: parseInt(process.env.SEARCH_TOP_K || "5"),
    vectorWeight: parseFloat(process.env.SEARCH_VECTOR_WEIGHT || "0.7"),
    fulltextWeight: parseFloat(process.env.SEARCH_FULLTEXT_WEIGHT || "0.3"),
  },

  graph: {
    similarityThreshold: parseFloat(process.env.GRAPH_SIMILARITY_THRESHOLD || "0.8"),
    similarityTopK: parseInt(process.env.GRAPH_SIMILARITY_TOP_K || "5"),
    minRelatedConnections: parseInt(process.env.GRAPH_MIN_RELATED_CONNECTIONS || "2"),
  },

  embeddingDimensions: {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-004": 768,
    "embedding-001": 768,
  } as Record<string, number>,
} as const;
