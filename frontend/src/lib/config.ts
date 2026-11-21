// Where the API lives (NEXT_PUBLIC_API_URL at build time; the local API by default).
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

/** An image address from the API: a path on the API (a Wikipedia poster from its cache, "/posters/1.webp")
 *  becomes a full URL; a full URL (a TMDB poster) is used as it is. */
export function assetUrl(url: string): string;
export function assetUrl(url: string | null | undefined): string | null;
export function assetUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  return url.startsWith("/") ? `${API_URL}${url}` : url;
}
