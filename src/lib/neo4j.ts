import neo4j, { type Driver, type Session, type ManagedTransaction } from "neo4j-driver";
import { config } from "../config/index.js";

let driver: Driver | null = null;

export function getNeo4jDriver(): Driver {
  if (!driver) {
    driver = neo4j.driver(
      config.neo4j.uri,
      neo4j.auth.basic(config.neo4j.username, config.neo4j.password),
    );
  }
  return driver;
}

export function getSession(): Session {
  return getNeo4jDriver().session({
    database: config.neo4j.database,
  });
}

/** Convert Neo4j Integer {low,high} objects to plain JS numbers recursively */
function toNative(val: unknown): unknown {
  if (val === null || val === undefined) return val;
  if (neo4j.isInt(val)) return val.toNumber();
  if (Array.isArray(val)) return val.map(toNative);
  if (typeof val === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(val as Record<string, unknown>)) {
      out[k] = toNative(v);
    }
    return out;
  }
  return val;
}

export async function runQuery<T = Record<string, unknown>>(
  cypher: string,
  params: Record<string, unknown> = {},
): Promise<T[]> {
  const session = getSession();
  try {
    const result = await session.run(cypher, params);
    return result.records.map((record) => toNative(record.toObject()) as T);
  } finally {
    await session.close();
  }
}

export async function runWriteTransaction<T>(
  work: (tx: ManagedTransaction) => Promise<T>,
): Promise<T> {
  const session = getSession();
  try {
    return await session.executeWrite(work);
  } finally {
    await session.close();
  }
}

export async function initNeo4j(): Promise<void> {
  const session = getSession();
  try {
    await session.run(
      "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.documentId IS UNIQUE",
    );
    await session.run(
      "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
    );
    await session.run(
      "CREATE INDEX chunk_document_id IF NOT EXISTS FOR (c:Chunk) ON (c.documentId)",
    );
    console.log("Neo4j constraints and indexes ensured");
  } finally {
    await session.close();
  }
}

export async function closeNeo4j(): Promise<void> {
  if (driver) {
    await driver.close();
    driver = null;
  }
}
