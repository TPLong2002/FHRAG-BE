import { Router } from "express";
import { chatStream } from "../services/chat.service.js";
import type { AuthRequest } from "../middleware/auth.js";
import type { LLMProvider } from "../types/index.js";

const router = Router();

/** Chat with SSE streaming */
router.post("/", async (req, res) => {
  try {
    const { question, provider, model, documentIds } = req.body;
    if (!question || !provider || !model) {
      res.status(400).json({ error: "question, provider, and model are required" });
      return;
    }

    const userId = (req as AuthRequest).userId;

    // Set up SSE
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    const sources = await chatStream(
      {
        question,
        provider: provider as LLMProvider,
        model,
        documentIds,
        userId,
      },
      (chunk) => {
        res.write(`data: ${JSON.stringify({ type: "chunk", content: chunk })}\n\n`);
      }
    );

    // Send sources at the end
    res.write(`data: ${JSON.stringify({ type: "sources", sources })}\n\n`);
    res.write("data: [DONE]\n\n");
    res.end();
  } catch (err) {
    console.error("Chat error:", err);
    if (!res.headersSent) {
      res.status(500).json({ error: (err as Error).message });
    } else {
      res.write(`data: ${JSON.stringify({ type: "error", error: (err as Error).message })}\n\n`);
      res.end();
    }
  }
});

export default router;
