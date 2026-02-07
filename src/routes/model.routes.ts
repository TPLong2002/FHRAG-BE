import { Router } from "express";
import { LLM_MODELS } from "../lib/llm.js";
import { EMBEDDING_MODELS } from "../lib/embeddings.js";

const router = Router();

/** List available LLM models */
router.get("/llm", (_req, res) => {
  res.json({ models: LLM_MODELS });
});

/** List available embedding models */
router.get("/embedding", (_req, res) => {
  res.json({ models: EMBEDDING_MODELS });
});

export default router;
