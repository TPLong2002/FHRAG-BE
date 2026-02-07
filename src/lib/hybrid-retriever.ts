import { BaseRetriever } from "@langchain/core/retrievers";
import { Document } from "@langchain/core/documents";
import type { Embeddings } from "@langchain/core/embeddings";
import type { Client } from "@opensearch-project/opensearch";
import type { AccessControl } from "../types/index.js";

export interface HybridRetrieverOptions {
  client: Client;
  embeddings: Embeddings;
  indexName: string;
  k?: number;
  pipeline?: string;
  /** Filter by specific document IDs */
  documentIds?: string[];
  /** For permission filtering: current user ID */
  userId?: string;
}

/**
 * OpenSearch hybrid retriever combining BM25 + kNN.
 * Supports document-level permission filtering via accessControl metadata.
 */
export class OpenSearchHybridRetriever extends BaseRetriever {
  lc_namespace = ["custom", "opensearch"];

  private client: Client;
  private embeddings: Embeddings;
  private indexName: string;
  private k: number;
  private pipeline: string;
  private documentIds?: string[];
  private userId?: string;

  constructor(fields: HybridRetrieverOptions) {
    super(fields);
    this.client = fields.client;
    this.embeddings = fields.embeddings;
    this.indexName = fields.indexName;
    this.k = fields.k ?? 4;
    this.pipeline = fields.pipeline ?? "hybrid-search-pipeline";
    this.documentIds = fields.documentIds;
    this.userId = fields.userId;
  }

  async _getRelevantDocuments(query: string): Promise<Document[]> {
    const queryVector = await this.embeddings.embedQuery(query);

    // Build permission filter
    const filters: Record<string, unknown>[] = [];

    // Filter by specific documents if provided
    if (this.documentIds?.length) {
      filters.push({ terms: { "metadata.documentId": this.documentIds } });
    }

    // Permission filter: public docs OR docs where user is allowed
    if (this.userId) {
      filters.push({
        bool: {
          should: [
            { term: { "metadata.accessControl.public": true } },
            { term: { "metadata.accessControl.allowedUsers": this.userId } },
            { term: { "metadata.ownerId": this.userId } },
          ],
          minimum_should_match: 1,
        },
      });
    }

    const hybridQuery: Record<string, unknown> = {
      hybrid: {
        queries: [
          { match: { text: { query } } },
          { knn: { embedding: { vector: queryVector, k: this.k } } },
        ],
      },
    };

    // Wrap with filter if needed
    const finalQuery = filters.length
      ? { bool: { must: [hybridQuery], filter: filters } }
      : hybridQuery;

    const response = await this.client.search({
      index: this.indexName,
      search_pipeline: this.pipeline,
      body: { size: this.k, query: finalQuery },
    });

    return response.body.hits.hits.map(
      (hit: Record<string, unknown>) => {
        const source = hit._source as Record<string, unknown>;
        const metadata = source.metadata as Record<string, unknown>;
        return new Document({
          pageContent: source.text as string,
          metadata: {
            ...metadata,
            _id: hit._id,
            _score: hit._score,
          },
        });
      }
    );
  }
}
