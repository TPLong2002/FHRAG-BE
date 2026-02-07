import { Client } from "@opensearch-project/opensearch";
import { config } from "../config/index.js";

export const osClient = new Client({
  nodes: [config.opensearch.url],
  auth: {
    username: config.opensearch.username,
    password: config.opensearch.password,
  },
  ssl: { rejectUnauthorized: false },
});

/** Ensure the search pipeline for hybrid search exists */
async function ensureSearchPipeline() {
  const pipelineName = config.opensearch.pipelineName;
  try {
    await osClient.transport.request({
      method: "GET",
      path: `/_search/pipeline/${pipelineName}`,
    });
  } catch {
    await osClient.transport.request({
      method: "PUT",
      path: `/_search/pipeline/${pipelineName}`,
      body: {
        description: "Hybrid search normalization pipeline",
        phase_results_processors: [
          {
            "normalization-processor": {
              normalization: { technique: "min_max" },
              combination: { technique: "arithmetic_mean", parameters: { weights: [0.3, 0.7] } },
            },
          },
        ],
      },
    });
    console.log(`Created search pipeline: ${pipelineName}`);
  }
}

/** Ensure the document chunks index exists with correct mappings */
async function ensureChunksIndex(dimension: number) {
  const indexName = config.opensearch.index;
  const exists = await osClient.indices.exists({ index: indexName });
  if (exists.body) return;

  await osClient.indices.create({
    index: indexName,
    body: {
      settings: {
        index: { knn: true },
      },
      mappings: {
        properties: {
          text: { type: "text", analyzer: "standard" },
          embedding: {
            type: "knn_vector",
            dimension,
            method: { name: "hnsw", space_type: "l2", engine: "lucene" },
          },
          metadata: {
            properties: {
              documentId: { type: "keyword" },
              fileName: { type: "keyword" },
              fileType: { type: "keyword" },
              chunkIndex: { type: "integer" },
              ownerId: { type: "keyword" },
              accessControl: {
                properties: {
                  public: { type: "boolean" },
                  allowedUsers: { type: "keyword" },
                  allowedGroups: { type: "keyword" },
                },
              },
            },
          },
        },
      },
    },
  });
  console.log(`Created index: ${indexName} (dim=${dimension})`);
}

/** Ensure the document metadata index exists */
async function ensureMetaIndex() {
  const indexName = config.opensearch.metaIndex;
  const exists = await osClient.indices.exists({ index: indexName });
  if (exists.body) return;

  await osClient.indices.create({
    index: indexName,
    body: {
      mappings: {
        properties: {
          fileName: { type: "keyword" },
          fileType: { type: "keyword" },
          fileSize: { type: "long" },
          totalChunks: { type: "integer" },
          ownerId: { type: "keyword" },
          accessControl: {
            properties: {
              public: { type: "boolean" },
              allowedUsers: { type: "keyword" },
              allowedGroups: { type: "keyword" },
            },
          },
          embeddingProvider: { type: "keyword" },
          embeddingModel: { type: "keyword" },
          uploadedAt: { type: "date" },
        },
      },
    },
  });
  console.log(`Created meta index: ${indexName}`);
}

export async function initOpenSearch(embeddingDimension: number) {
  await ensureSearchPipeline();
  await ensureChunksIndex(embeddingDimension);
  await ensureMetaIndex();
}
