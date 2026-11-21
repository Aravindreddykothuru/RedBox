"use client";

import { API_URL } from "./config";
import { currentSession, endSession } from "./session";

export { API_URL };

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Query = Record<string, string | number | boolean | string[] | null | undefined>;

function withQuery(path: string, query?: Query) {
  if (!query) return path;
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === null || v === undefined || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => params.append(k, x));
    else params.set(k, String(v));
  }
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

export async function api<T>(
  path: string,
  options: { method?: string; query?: Query; body?: unknown; signal?: AbortSignal; text?: boolean } = {},
): Promise<T> {
  const session = currentSession();
  const headers: Record<string, string> = {};
  if ((session?.mode === "supabase" || session?.mode === "local") && session.accessToken) {
    headers.Authorization = `Bearer ${session.accessToken}`;
  } else if (session?.mode === "demo") {
    headers["X-User-Id"] = session.userId;
  }
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  let res: Response;
  try {
    res = await fetch(`${API_URL}${withQuery(path, options.query)}`, {
      method: options.method ?? "GET",
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options.signal,
    });
  } catch {
    throw new ApiError(0, "Can't reach the recommendation service. Is the API running?");
  }
  if (res.status === 401 && session?.mode === "local" && session.accessToken) {
    endSession(); // the token expired or was revoked (password change, "sign out everywhere")
    throw new ApiError(401, "Your session ended. Please sign in again.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // not JSON
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204 || res.status === 202) return undefined as T;
  if (options.text) return (await res.text()) as T; // e.g. CSV downloads
  return (await res.json()) as T;
}

/** Fire-and-forget interaction logging (clicks), never blocks the UI. */
export function logEvent(movieId: number, type: "click" | "impression", sourceModel?: string, position?: number) {
  api("/events", {
    method: "POST",
    body: { movie_id: movieId, type, source_model: sourceModel ?? null, position: position ?? null },
  }).catch(() => undefined);
}
