import express from "express";
import cors from "cors";
import { config } from "./config/index.js";
import { authMiddleware } from "./middleware/auth.js";
import { initNeo4j, closeNeo4j } from "./lib/neo4j.js";
import documentRoutes from "./routes/document.routes.js";
import chatRoutes from "./routes/chat.routes.js";
import modelRoutes from "./routes/model.routes.js";
import graphRoutes from "./routes/graph.routes.js";

const app = express();

app.use(cors());
app.use(express.json());
app.use(authMiddleware);

app.use("/api/documents", documentRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/models", modelRoutes);
app.use("/api/graph", graphRoutes);

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.listen(config.port, async () => {
  console.log(`Backend running on http://localhost:${config.port}`);
  try {
    await initNeo4j();
  } catch (err) {
    console.error("Neo4j init failed (graph features disabled):", err);
  }
});

process.on("SIGINT", async () => {
  await closeNeo4j();
  process.exit(0);
});

process.on("SIGTERM", async () => {
  await closeNeo4j();
  process.exit(0);
});
