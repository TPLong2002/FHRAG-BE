import { Router } from "express";
import multer from "multer";
import path from "path";
import fs from "fs/promises";
import { uploadDocument, listDocuments, deleteDocument } from "../services/document.service.js";
import type { AuthRequest } from "../middleware/auth.js";
import type { EmbeddingProvider } from "../types/index.js";

const router = Router();

const upload = multer({
  dest: "uploads/",
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
  fileFilter: (_req, file, cb) => {
    const allowed = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
      "text/csv",
      "application/csv",
      "text/plain",
    ];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type: ${file.mimetype}`));
    }
  },
});

/** Upload one or more files */
router.post("/upload", upload.array("files", 10), async (req, res) => {
  try {
    const files = req.files as Express.Multer.File[];
    if (!files?.length) {
      res.status(400).json({ error: "No files provided" });
      return;
    }

    const embeddingProvider = (req.body.embeddingProvider || "openai") as EmbeddingProvider;
    const embeddingModel = req.body.embeddingModel || "text-embedding-3-small";
    const userId = (req as AuthRequest).userId;

    const results = [];
    for (const file of files) {
      // Multer decodes filename as latin1 — re-encode to get correct UTF-8
      const fileName = Buffer.from(file.originalname, "latin1").toString("utf8");
      const meta = await uploadDocument(file.path, fileName, file.mimetype, file.size, {
        embeddingProvider,
        embeddingModel,
        ownerId: userId,
      });
      results.push(meta);

      // Clean up uploaded file
      await fs.unlink(file.path).catch(() => {});
    }

    res.json({ documents: results });
  } catch (err) {
    console.error("Upload error:", err);
    res.status(500).json({ error: (err as Error).message });
  }
});

/** List all documents */
router.get("/", async (req, res) => {
  try {
    const userId = (req as AuthRequest).userId;
    const docs = await listDocuments(userId);
    res.json({ documents: docs });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

/** Delete a document */
router.delete("/:id", async (req, res) => {
  try {
    await deleteDocument(req.params.id);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

export default router;
