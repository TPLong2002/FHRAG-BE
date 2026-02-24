import { Router } from "express";
import {
  getRelatedDocuments,
  getDocumentGraph,
  getChunkGraph,
  getSchemaGraph,
  deleteTable,
  deleteForeignKey,
} from "../services/graph.service.js";

const router = Router();

/** Get document-level graph overview */
router.get("/documents", async (_req, res) => {
  try {
    const graph = await getDocumentGraph();
    res.json(graph);
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/** Get single document graph (with related docs) */
router.get("/documents/:id", async (req, res) => {
  try {
    const graph = await getDocumentGraph(req.params.id);
    res.json(graph);
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/** Get related documents for a specific document */
router.get("/documents/:id/related", async (req, res) => {
  try {
    const related = await getRelatedDocuments(req.params.id);
    res.json({ related });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/** Get chunk-level graph for a document */
router.get("/documents/:id/chunks", async (req, res) => {
  try {
    const graph = await getChunkGraph(req.params.id);
    res.json(graph);
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/** Get schema graph (tables + foreign keys) */
router.get("/schema", async (req, res) => {
  try {
    const documentId = req.query.documentId as string | undefined;
    const graph = await getSchemaGraph(documentId);
    res.json(graph);
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/** Delete a table and all its relationships */
router.delete("/schema/tables/:name", async (req, res) => {
  try {
    await deleteTable(req.params.name);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/** Delete a foreign key relationship */
router.delete("/schema/foreign-keys", async (req, res) => {
  try {
    const { fromTable, toTable, fromColumn, toColumn } = req.body;
    if (!fromTable || !toTable || !fromColumn || !toColumn) {
      res.status(400).json({ error: "Missing required fields: fromTable, toTable, fromColumn, toColumn" });
      return;
    }
    await deleteForeignKey(fromTable, toTable, fromColumn, toColumn);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

export default router;
