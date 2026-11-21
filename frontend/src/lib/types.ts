// Mirrors the API's Pydantic schemas (backend/app/schemas.py).

export type Movie = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  popularity: number | null;
  poster_url: string | null;
};

export type MovieDetail = Movie & {
  overview: string | null;
  overview_source?: string | null; // "wikipedia" or "tmdb"
  language?: string | null; // new films outside MovieLens, e.g. "Telugu"
  tagline: string | null;
  runtime: number | null;
  cast: string[];
  keywords: string[];
  avg_rating: number | null;
  n_ratings: number | null;
  tmdb_id: number | null;
  imdb_id: string | null;
  your_rating: number | null;
  in_watchlist: boolean;
};

export type Recommendation = Movie & { score: number; reason: string };

export type RatedMovie = Movie & { rating: number; rated_at: string };

export type MoviePage = { total: number; items: Movie[] };

export type FeedRow = { key: string; title: string; items: Recommendation[] };

export type Feed = { rows: FeedRow[] };

export type Profile = {
  display_name: string | null;
  favorite_genres: string[];
  onboarded: boolean;
  n_ratings: number;
  n_watched: number; // films from an imported watch history
};

// "Why this pick?": the ranker's score for one film, split by signal (TreeSHAP)
export type WhyOut = {
  available: boolean;
  rank: number | null;
  candidates: number | null;
  proposed_by: string[];
  parts: { signal: string; contribution: number; share: number }[];
};

// Watch-history import (Netflix, IMDb, Letterboxd): /me/import/preview, then /me/import
export type ImportSource = "netflix" | "imdb" | "letterboxd";
export type ImportKind = "rating" | "watched" | "watchlist";
export type ImportMatch = {
  movie: Movie;
  kind: ImportKind;
  rating: number | null;
  at: string | null;
  source_title: string; // the title as the other service wrote it
  times: number; // viewings or ratings (rewatches)
  already: boolean; // already rated or imported in Red Box
};
export type ImportPreview = {
  source: ImportSource;
  rows: number;
  tv_skipped: number;
  short_skipped: number;
  unmatched_count: number;
  unmatched_sample: string[];
  items: ImportMatch[];
};
export type ImportResult = { ratings: number; watched: number; watchlist: number; unchanged: number };
export type WatchedMovie = Movie & { watched_at: string; source: ImportSource; times: number };

export type Health = {
  status: string;
  dataset: string;
  bundle_created: string;
  n_items: number;
  models: string[];
  semantic_search: boolean;
  serving_test_metrics: Record<string, number>;
};

export type LabUser = { user_id: number; n_liked: number };

export type LabCompare = { history: Movie[]; models: Record<string, Recommendation[]> };

export type OfflineRow = Record<string, string | number | null>;

export type LabMetrics = {
  serving_model: Record<string, number>;
  offline: OfflineRow[];
};
