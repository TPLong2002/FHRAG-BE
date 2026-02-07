import express from "express";
import cors from "cors";
import { config } from "./config/index.js";
import { authMiddleware } from "./middleware/auth.js";
import documentRoutes from "./routes/document.routes.js";
import chatRoutes from "./routes/chat.routes.js";
import modelRoutes from "./routes/model.routes.js";

const app = express();

app.use(cors());
app.use(express.json());
app.use(authMiddleware);

app.use("/api/documents", documentRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/models", modelRoutes);

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.listen(config.port, () => {
  console.log(`Backend running on http://localhost:${config.port}`);
});
