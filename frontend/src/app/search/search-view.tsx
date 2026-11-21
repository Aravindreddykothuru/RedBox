"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { MovieCard, MovieGrid } from "@/components/movie-card";
import { Chip, EmptyState, ErrorState, PageHeader, Segmented, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { useHealth } from "@/lib/hooks";
import type { MoviePage, Recommendation } from "@/lib/types";

type Mode = "describe" | "title";

const EXAMPLES = [
  "mind-bending films about dreams",
  "feel-good animated movie for kids",
  "time travel",
  "romantic comedy in new york",
  "dark dystopian future",
  "heist with a clever twist",
];

export function SearchView() {
  const { t } = useI18n();
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const health = useHealth();
  const semanticAvailable = health.data?.semantic_search ?? true;
  const mode: Mode = params.get("mode") === "title" || !semanticAvailable ? "title" : "describe";
  const urlQuery = params.get("q") ?? "";
  const genre = params.get("genre") ?? "";
  const [text, setText] = useState(urlQuery);

  // keep the URL (shareable, back button) in sync with the box, 350 ms after typing stops
  useEffect(() => {
    if (text === urlQuery) return;
    const t = window.setTimeout(() => update({ q: text }), 350);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  function update(changes: Record<string, string>) {
    const next = new URLSearchParams(params.toString());
    for (const [k, v] of Object.entries(changes)) {
      if (v) next.set(k, v);
      else next.delete(k);
    }
    router.replace(`${pathname}?${next.toString()}`, { scroll: false });
  }

  const genres = useQuery({ queryKey: ["genres"], queryFn: () => api<string[]>("/movies/genres") });
  const semantic = useQuery({
    queryKey: ["semantic", urlQuery],
    queryFn: () => api<Recommendation[]>("/search/semantic", { query: { q: urlQuery, k: 36 } }),
    enabled: mode === "describe" && urlQuery.trim().length >= 2,
    placeholderData: keepPreviousData,
  });
  const titles = useQuery({
    queryKey: ["titles", urlQuery, genre],
    queryFn: () => api<MoviePage>("/movies", { query: { q: urlQuery, genre, limit: 48 } }),
    enabled: mode === "title",
    placeholderData: keepPreviousData,
  });

  return (
    <div>
      <PageHeader title={t("nav.search")}>{mode === "describe" ? t("search.describeBody") : t("search.titleBody")}</PageHeader>

      <div className="flex flex-wrap items-center gap-3">
        <Segmented
          label="Search mode"
          value={mode}
          onChange={(v) => update({ mode: v === "title" ? "title" : "" })}
          options={[
            ...(semanticAvailable ? [{ value: "describe" as Mode, label: t("search.describe") }] : []),
            { value: "title", label: t("search.byTitle") },
          ]}
        />
        {mode === "title" ? (
          <select
            aria-label="Genre"
            value={genre}
            onChange={(e) => update({ genre: e.target.value })}
            className="h-9 rounded-lg border border-line bg-surface px-3 text-sm text-ink"
          >
            <option value="">All genres</option>
            {genres.data?.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        ) : null}
      </div>

      <form
        role="search"
        className="relative mt-4 max-w-2xl"
        onSubmit={(e) => {
          e.preventDefault();
          update({ q: text });
        }}
      >
        <Search size={16} strokeWidth={1.5} className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-muted" aria-hidden />
        <input
          type="search"
          aria-label={mode === "describe" ? "Describe a film" : "Movie title"}
          value={text}
          maxLength={mode === "describe" ? 200 : 100}
          onChange={(e) => setText(e.target.value)}
          placeholder={mode === "describe" ? t("search.phDescribe") : t("search.phTitle")}
          className="h-12 w-full rounded border border-line-strong bg-black/60 pl-10 pr-4 text-base text-ink placeholder:text-muted focus:border-ink"
        />
      </form>

      {mode === "describe" && !urlQuery ? (
        <div className="mt-4 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <Chip
              key={ex}
              onClick={() => {
                setText(ex);
                update({ q: ex });
              }}
            >
              {ex}
            </Chip>
          ))}
        </div>
      ) : null}

      <div className="mt-8">
        {mode === "describe" ? (
          <Results
            query={urlQuery}
            isPending={semantic.isFetching && !semantic.data}
            error={semantic.error}
            items={urlQuery.trim().length >= 2 ? semantic.data : undefined}
            dim={semantic.isPlaceholderData}
            source="search"
          />
        ) : (
          <>
            {titles.data ? (
              <p className="mb-4 text-sm text-muted">
                {titles.data.total.toLocaleString()} film{titles.data.total === 1 ? "" : "s"}
                {titles.data.total > 48 ? ", most watched first" : ""}
              </p>
            ) : null}
            <Results
              query={urlQuery || genre || "all"}
              isPending={titles.isPending}
              error={titles.error}
              items={titles.data?.items}
              dim={titles.isPlaceholderData}
            />
          </>
        )}
      </div>
    </div>
  );
}

function Results({
  query,
  isPending,
  error,
  items,
  dim,
  source,
}: {
  query: string;
  isPending: boolean;
  error: unknown;
  items: (Recommendation | MoviePage["items"][number])[] | undefined;
  dim?: boolean;
  source?: string;
}) {
  if (error) return <ErrorState error={error} />;
  if (isPending)
    return (
      <MovieGrid>
        {Array.from({ length: 12 }, (_, i) => (
          <Skeleton key={i} className="aspect-[2/3] w-full" />
        ))}
      </MovieGrid>
    );
  if (!items) return null;
  if (!items.length) return <EmptyState title={`Nothing found for “${query}”`}>Try fewer or different words.</EmptyState>;
  return (
    <div className={dim ? "opacity-60 transition-opacity" : "transition-opacity"}>
      <MovieGrid>
        {items.map((m, i) => (
          <MovieCard key={m.movie_id} movie={m} source={source} position={i} width="w-full" />
        ))}
      </MovieGrid>
    </div>
  );
}
