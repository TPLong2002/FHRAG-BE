import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createLLM } from "../lib/llm.js";
import { runQuery } from "../lib/neo4j.js";
import type { LLMProvider } from "../types/index.js";
import { config } from "../config/index.js";

// ==================== Zod Schema ====================

const ColumnSchema = z.object({
  name: z.string().describe("Column name"),
  type: z.string().describe("Data type (e.g., VARCHAR, INT, TIMESTAMP)"),
  nullable: z.boolean().describe("Whether the column allows NULL"),
  isPrimaryKey: z.boolean().describe("Whether this is a primary key"),
});

const ForeignKeySchema = z.object({
  fromTable: z.string().describe("Source table name"),
  fromColumn: z.string().describe("Source column name"),
  toTable: z.string().describe("Referenced table name"),
  toColumn: z.string().describe("Referenced column name"),
});

const TableSchema = z.object({
  name: z.string().describe("Table name"),
  description: z.string().describe("Brief description of what this table stores"),
  columns: z.array(ColumnSchema).describe("List of columns"),
});

const SchemaExtractionResult = z.object({
  tables: z.array(TableSchema).describe("Database tables found in this text"),
  foreignKeys: z.array(ForeignKeySchema).describe("Foreign key relationships"),
});

type SchemaExtractionResult = z.infer<typeof SchemaExtractionResult>;

// ==================== LLM Prompt ====================

const EXTRACTION_PROMPT = `You are a database schema analyst. Extract database table definitions and relationships from the given text.

Valid table definitions include any of these formats:
1. DBML: Table Name with columns and their types inside braces
2. SQL: CREATE TABLE Name (ColumnName TYPE constraints, ...)
3. Structured lists of columns with explicit data types (int, nvarchar, varchar, datetime, bit, float, etc.)

Example DBML:
  Table DM_NhanVien
    ID_NhanVien int [pk, not null]
    MaNV nvarchar(50)
    ID_DonVi int

Rules:
- Extract tables ONLY when columns with data types are present in the text.
- Each column belongs ONLY to the table it is listed under. Do NOT assign columns from one table to another.
- If the text contains multiple tables, extract each separately with its own columns.
- Do NOT extract from plain text descriptions like "the employee table stores..." without column definitions.
- Do NOT invent tables or columns not in the text.
- If no structured table definitions exist, return empty arrays.
- Normalize table names to their most common form.
- For foreign keys, only include relationships explicitly stated.
- Set isPrimaryKey to true only if indicated (e.g., [pk], PRIMARY KEY).
- Set nullable to false if [not null]/NOT NULL is specified, otherwise default to true.
- Include column data types as written. Use "UNKNOWN" only if truly unspecified.`;

const MERGE_PROMPT = `You are a database schema analyst. You are given two versions of the same table extracted from different parts of a document.
Merge them into a single, complete table definition.

Rules:
- Combine all unique columns from both versions.
- If the same column appears in both, keep the one with more complete information (type, nullable, isPrimaryKey).
- Choose the longer/more descriptive description.
- Do NOT invent new columns or modify existing ones beyond merging.
- The result must contain ONLY information present in the two input versions.`;

const extractionPrompt = ChatPromptTemplate.fromMessages([
  ["system", EXTRACTION_PROMPT],
  ["human", "{text}"],
]);

const mergePrompt = ChatPromptTemplate.fromMessages([
  ["system", MERGE_PROMPT],
  ["human", "Table name: {tableName}\n\nVersion 1:\n{version1}\n\nVersion 2:\n{version2}"],
]);

// ==================== Merge Logic ====================

interface MergedTable {
  name: string;
  displayName: string;
  description: string;
  columns: z.infer<typeof ColumnSchema>[];
}

function collectExtractionResults(results: SchemaExtractionResult[]) {
  // Keep only the best version per table (most columns)
  const tables = new Map<string, MergedTable>();
  const allForeignKeys: z.infer<typeof ForeignKeySchema>[] = [];
  const tableChunkMap = new Map<string, number[]>();

  for (let i = 0; i < results.length; i++) {
    const result = results[i];

    for (const table of result.tables) {
      const normalizedName = table.name.toLowerCase().trim();

      const indices = tableChunkMap.get(normalizedName) || [];
      indices.push(i);
      tableChunkMap.set(normalizedName, indices);

      const existing = tables.get(normalizedName);
      if (!existing || table.columns.length > existing.columns.length) {
        tables.set(normalizedName, {
          name: normalizedName,
          displayName: table.name,
          description: table.description,
          columns: [...table.columns],
        });
      }
    }

    allForeignKeys.push(...result.foreignKeys);
  }

  // Deduplicate foreign keys
  const fkSet = new Set<string>();
  const uniqueFKs = allForeignKeys.filter((fk) => {
    const key = `${fk.fromTable.toLowerCase()}.${fk.fromColumn.toLowerCase()}->${fk.toTable.toLowerCase()}.${fk.toColumn.toLowerCase()}`;
    if (fkSet.has(key)) return false;
    fkSet.add(key);
    return true;
  });

  return { tables, foreignKeys: uniqueFKs, tableChunkMap };
}

// ==================== Main Entry Point ====================

export async function extractAndStoreSchema(
  documentId: string,
  chunks: string[],
  chunkIds: string[],
): Promise<void> {
  try {
    // Determine cheap LLM based on available API keys
    let provider: LLMProvider;
    let model: string;
    // if (config.apiKeys.openai) {
    //   provider = "openai";
    //   model = "gpt-4.1-mini";
    // } else 
    if (config.apiKeys.google) {
      provider = "google";
      model = "gemini-3-flash-preview";
    } else {
      console.warn("No LLM API key available for schema extraction");
      return;
    }
    console.log(`Starting schema extraction for document ${documentId} using ${provider} (${model})`);

    const llm = createLLM(provider, model);
    const structured = llm.withStructuredOutput(SchemaExtractionResult);
    const chain = extractionPrompt.pipe(structured);

    // Check if schema already exists for this document
    const existingSchema = await runQuery<{ tableCount: number }>(
      `MATCH (d:Document {documentId: $documentId})-[:HAS_TABLE]->(t:Table)
      RETURN count(t) AS tableCount`,
      { documentId },
    );
    if (existingSchema[0]?.tableCount > 0) {
      console.log(`Schema already exists for document ${documentId}, skipping extraction`);
      return;
    }

    // Build merged chunks: combine each chunk with up to 2 adjacent chunks
    // to avoid missing schemas that span chunk boundaries
    const mergedChunks: string[] = [];
    for (let i = 0; i < chunks.length; i++) {
      const start = Math.max(0, i - 1);
      const end = Math.min(chunks.length, i + 2); // i-1, i, i+1
      mergedChunks.push(chunks.slice(start, end).join("\n"));
    }

    // Extract from each merged chunk in batches
    const results: SchemaExtractionResult[] = [];
    const batchSize = 5;
    console.log(`Extracting schema from ${mergedChunks.length} merged chunks in batches of ${batchSize}`);
    for (let i = 0; i < mergedChunks.length; i += batchSize) {
      const batch = mergedChunks.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map((text) =>
          chain.invoke({ text }).then((r) => r as SchemaExtractionResult).catch((err) => {
            console.error(`Schema extraction failed for chunk:`, err?.message || err);
            return { tables: [], foreignKeys: [] } as SchemaExtractionResult;
          }),
        ),
      );
      results.push(...batchResults);
      console.log(`Completed batch ${i / batchSize + 1}/${Math.ceil(mergedChunks.length / batchSize)}`);
    }

    // Collect + deduplicate (keep best version per table)
    console.log(`[Schema] Collecting and deduplicating extraction results...`);
    const { tables, foreignKeys, tableChunkMap } = collectExtractionResults(results);

    if (tables.size === 0) {
      console.log(`[Schema] No tables extracted from document ${documentId}`);
      return;
    }

    console.log(`[Schema] Found ${tables.size} unique tables, checking DB for existing versions...`);

    // LLM merge with DB only if table already exists
    const mergeStructured = llm.withStructuredOutput(SchemaExtractionResult);
    const mergeChain = mergePrompt.pipe(mergeStructured);
    let dbMergeCount = 0;

    for (const [normalizedName, table] of tables) {
      const existingRows = await runQuery<{ description: string; columns: string }>(
        `MATCH (t:Table {name: $name}) RETURN t.description AS description, t.columns AS columns`,
        { name: normalizedName },
      );
      if (existingRows.length > 0 && existingRows[0].columns) {
        try {
          const existingColumns = JSON.parse(existingRows[0].columns);
          console.log(`[Schema] Table "${normalizedName}" exists in DB (${existingColumns.length} cols), merging with extracted (${table.columns.length} cols)...`);
          const result = await mergeChain.invoke({
            tableName: table.displayName,
            version1: JSON.stringify({ description: existingRows[0].description, columns: existingColumns }, null, 2),
            version2: JSON.stringify({ description: table.description, columns: table.columns }, null, 2),
          });
          const parsed = result as SchemaExtractionResult;
          if (parsed.tables.length > 0) {
            table.description = parsed.tables[0].description;
            table.columns = parsed.tables[0].columns;
            console.log(`[Schema] Merged "${normalizedName}": ${table.columns.length} columns`);
          }
          dbMergeCount++;
        } catch (err) {
          console.error(`[Schema] Merge failed for "${normalizedName}":`, (err as Error)?.message || err);
        }
      }
    }
    console.log(`[Schema] ${dbMergeCount} tables merged with DB`);

    console.log(`[Schema] Final: ${tables.size} tables, ${foreignKeys.length} FKs from document ${documentId}`);

    // Store in Neo4j
    console.log(`[Schema] Storing tables in Neo4j...`);
    for (const [normalizedName, table] of tables) {
      console.log(`[Schema] Saving table "${normalizedName}" (${table.columns.length} columns)`);
      await runQuery(
        `MERGE (t:Table {name: $name})
        ON CREATE SET t.displayName = $displayName,
                      t.description = $description,
                      t.columns = $columns,
                      t.createdAt = $updatedAt,
                      t.updatedAt = $updatedAt
        ON MATCH SET t.displayName = $displayName,
                     t.description = $description,
                     t.columns = $columns,
                     t.updatedAt = $updatedAt`,
        {
          name: normalizedName,
          displayName: table.displayName,
          description: table.description,
          columns: JSON.stringify(table.columns),
          updatedAt: new Date().toISOString(),
        },
      );

      // HAS_TABLE relationship
      await runQuery(
        `MATCH (d:Document {documentId: $documentId})
        MATCH (t:Table {name: $tableName})
        MERGE (d)-[:HAS_TABLE]->(t)`,
        { documentId, tableName: normalizedName },
      );

      // MENTIONS_TABLE from relevant chunks
      const mentionIndices = tableChunkMap.get(normalizedName) || [];
      const mentionChunkIds = mentionIndices.map((idx: number) => chunkIds[idx]).filter(Boolean);
      if (mentionChunkIds.length > 0) {
        await runQuery(
          `MATCH (t:Table {name: $tableName})
          UNWIND $chunkIds AS cid
          MATCH (c:Chunk {chunkId: cid})
          MERGE (c)-[:MENTIONS_TABLE]->(t)`,
          { tableName: normalizedName, chunkIds: mentionChunkIds },
        );
      }
    }

    // Create FOREIGN_KEY relationships
    if (foreignKeys.length > 0) {
      console.log(`[Schema] Creating ${foreignKeys.length} foreign key relationships...`);
    }
    for (const fk of foreignKeys) {
      await runQuery(
        `MATCH (from:Table {name: $fromTable})
        MATCH (to:Table {name: $toTable})
        MERGE (from)-[r:FOREIGN_KEY {fromColumn: $fromColumn, toColumn: $toColumn}]->(to)`,
        {
          fromTable: fk.fromTable.toLowerCase().trim(),
          toTable: fk.toTable.toLowerCase().trim(),
          fromColumn: fk.fromColumn,
          toColumn: fk.toColumn,
        },
      );
    }

    console.log(`[Schema] Done for document ${documentId}`);

  } catch (error) {
    console.error(`Error during schema extraction for document ${documentId}:`, error);
  }
}
