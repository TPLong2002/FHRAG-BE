export type LLMProvider = "openai" | "google";
export type EmbeddingProvider = "openai" | "google";

export interface AccessControl {
  public: boolean;
  allowedUsers: string[];
  allowedGroups: string[];
}

export interface DocumentMeta {
  id: string;
  fileName: string;
  fileType: string;
  fileSize: number;
  totalChunks: number;
  ownerId: string;
  accessControl: AccessControl;
  embeddingProvider: EmbeddingProvider;
  embeddingModel: string;
  uploadedAt: string;
}

export interface ChunkMetadata {
  documentId: string;
  fileName: string;
  fileType: string;
  chunkIndex: number;
  ownerId: string;
  accessControl: AccessControl;
}

export interface ChatRequest {
  question: string;
  provider: LLMProvider;
  model: string;
  documentIds?: string[];
  userId?: string;
}

export interface ChatSource {
  documentId: string;
  fileName: string;
  chunkIndex: number;
  content: string;
  score: number;
}

export interface ModelInfo {
  provider: LLMProvider;
  id: string;
  name: string;
}

export interface UploadOptions {
  embeddingProvider: EmbeddingProvider;
  embeddingModel: string;
  ownerId?: string;
}
