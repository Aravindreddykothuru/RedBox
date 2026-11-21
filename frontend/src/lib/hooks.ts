"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { useSession } from "./session";
import type { Health, Profile, RatedMovie } from "./types";

/** Query key prefix that changes with the signed-in user, so switching accounts never shows stale data. */
export function useUserKey(): string {
  const { session } = useSession();
  return session?.userId ?? "anonymous";
}

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: () => api<Health>("/health"), staleTime: 300_000 });
}

export function useProfile() {
  const { session } = useSession();
  return useQuery({
    queryKey: ["profile", session?.userId],
    queryFn: () => api<Profile>("/me/profile"),
    enabled: !!session,
  });
}

export function useMyRatings() {
  const { session } = useSession();
  return useQuery({
    queryKey: ["my-ratings", session?.userId],
    queryFn: () => api<RatedMovie[]>("/me/ratings"),
    enabled: !!session,
  });
}

/** Everything personal changes after a rating: recommendations, feed, lab, profile, ratings. */
function invalidatePersonal(qc: ReturnType<typeof useQueryClient>) {
  for (const key of ["recs", "feed", "lab", "profile", "my-ratings", "movie", "seeds", "why", "new-films"]) {
    qc.invalidateQueries({ queryKey: [key] });
  }
}

export function useRate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ movieId, rating }: { movieId: number; rating: number }) =>
      api<RatedMovie>(`/ratings/${movieId}`, { method: "PUT", body: { rating } }),
    onSuccess: () => invalidatePersonal(qc),
  });
}

export function useUnrate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (movieId: number) => api<void>(`/ratings/${movieId}`, { method: "DELETE" }),
    onSuccess: () => invalidatePersonal(qc),
  });
}

export function useWatchlistToggle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ movieId, add }: { movieId: number; add: boolean }) =>
      api<void>(`/watchlist/${movieId}`, { method: add ? "PUT" : "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["watchlist"] });
      qc.invalidateQueries({ queryKey: ["movie"] });
    },
  });
}
