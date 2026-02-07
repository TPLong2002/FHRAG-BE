import type { Request, Response, NextFunction } from "express";

/**
 * Auth middleware placeholder.
 * Currently extracts userId from x-user-id header.
 * Replace with JWT/session-based auth when implementing real authorization.
 */
export function authMiddleware(req: Request, _res: Response, next: NextFunction) {
  const userId = req.headers["x-user-id"] as string | undefined;
  if (userId) {
    (req as Request & { userId?: string }).userId = userId;
  }
  next();
}

/** Typed request with optional userId */
export interface AuthRequest extends Request {
  userId?: string;
}
