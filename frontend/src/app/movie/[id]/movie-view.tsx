"use client";

import { useQuery } from "@tanstack/react-query";
import { Bookmark, BookmarkCheck, ExternalLink } from "lucide-react";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Backdrop } from "@/components/backdrop";
import { MovieRow, RowSkeleton } from "@/components/movie-row";
import { Poster } from "@/components/poster";
import { StarRating } from "@/components/star-rating";
import { WhyBreakdown } from "@/components/why-breakdown";
import { Button, ErrorState, Segmented } from "@/components/ui";
import { api } from "@/lib/api";
import { useRate, useUnrate, useWatchlistToggle } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import { modelName } from "@/lib/models";
import { useSession } from "@/lib/session";
import type { MovieDetail, Recommendation } from "@/lib/types";

export function MovieView() {
  const { id } = useParams<{ id: string }>();
  const search = useSearchParams();
  const why = search.get("why");
  const src = search.get("src");
  const { session } = useSession();
  const movieId = Number(id);
  const { t, genre, reason } = useI18n();
  const detail = useQuery({
    queryKey: ["movie", movieId, session?.userId ?? "anonymous"],
    queryFn: () => api<MovieDetail>(`/movies/${movieId}`),
    enabled: Number.isInteger(movieId),
  });

  if (detail.isError) return <div className="pt-10"><ErrorState error={detail.error} retry={() => detail.refetch()} /></div>;
  if (!detail.data) return <RowSkeleton />;
  const m = detail.data;

  return (
    <div>
      <section className="relative isolate -mt-16 ml-[calc(50%-50vw)] w-screen">
        <Backdrop variant="detail" image={m.poster_url} />
        <div className="enter mx-auto grid max-w-[1600px] gap-8 px-4 pb-10 pt-28 sm:px-8 md:grid-cols-[260px_1fr] lg:gap-12 lg:px-12">
        <div className="mx-auto w-56 md:w-full">
          <Poster movie={m} sizes="lg" className="shadow-[0_16px_40px_rgb(0_0_0/0.6)]" />
        </div>
        <div>
          <h1 className="text-[36px] font-extrabold leading-10 tracking-tight sm:text-[48px] sm:leading-[52px]" style={{ textWrap: "balance" }}>
            {m.title}
          </h1>
          <p className="mt-3 text-base text-ink-2">
            {[m.year, m.runtime ? formatRuntime(m.runtime) : null].filter(Boolean).join("  ·  ")}
          </p>
          {m.avg_rating !== null && m.n_ratings ? (
            <p className="mt-1 text-[13px] text-muted">
              <span className="text-accent-text" aria-hidden>
                ★
              </span>{" "}
              <span className="tabular-nums">{m.avg_rating.toFixed(1)}</span>{" "}
              {t("movie.avgFrom", { n: m.n_ratings.toLocaleString() })}
            </p>
          ) : null}
          {m.tagline ? <p className="mt-4 text-lg font-medium text-ink-2">{m.tagline}</p> : null}
          {m.overview ? <p className="mt-4 max-w-2xl text-base leading-6">{m.overview}</p> : null}

          {why ? (
            <div className="mt-6 max-w-2xl rounded border-l-[3px] border-accent bg-black/50 px-4 py-3">
              <p className="text-[13px] font-semibold text-accent-text">{t("movie.why")}</p>
              <p className="mt-1 text-base text-ink">
                {reason(why)}
                {src && src !== "trending" && src !== "new_films" ? ` · ${ROW_SOURCES[src] ?? modelName(src)}` : ""}
              </p>
            </div>
          ) : null}
          {session ? <WhyBreakdown movieId={m.movie_id} userId={session.userId} /> : null}

          <div className="mt-6 flex flex-wrap items-center gap-6">
            {session ? <RatingControl movie={m} /> : (
              <Link href={`/login?next=/movie/${m.movie_id}`} className="text-ink underline underline-offset-4 hover:text-white">
                {t("movie.signInToRate")}
              </Link>
            )}
            {session ? <WatchlistButton movie={m} /> : null}
          </div>

          {m.keywords.length ? (
            <div className="mt-8">
              <h2 className="text-[13px] text-muted">{t("movie.describe")}</h2>
              <ul className="mt-2 flex flex-wrap gap-2">
                {m.keywords.map((k) => (
                  <li key={k}>
                    <Link
                      href={`/search?mode=describe&q=${encodeURIComponent(k)}`}
                      className="inline-flex h-8 items-center rounded border border-line-strong bg-black/30 px-3 text-[13px] text-ink-2 transition-colors duration-200 hover:border-ink-2 hover:text-ink"
                    >
                      {k}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <dl className="mt-6 space-y-1 text-sm">
            {m.cast.length ? (
              <div>
                <dt className="inline text-muted">{t("movie.cast")} </dt>
                <dd className="inline text-ink">{m.cast.slice(0, 5).join(", ")}</dd>
              </div>
            ) : null}
            {m.genres.length ? (
              <div>
                <dt className="inline text-muted">{t("movie.genres")} </dt>
                <dd className="inline text-ink">{m.genres.map(genre).join(", ")}</dd>
              </div>
            ) : null}
          </dl>

          <div className="mt-8 flex flex-wrap gap-4 text-[13px]">
            {m.tmdb_id ? (
              <a className="inline-flex items-center gap-1 text-muted hover:text-ink" href={`https://www.themoviedb.org/movie/${m.tmdb_id}`} target="_blank" rel="noreferrer">
                TMDB <ExternalLink size={14} strokeWidth={1.5} />
              </a>
            ) : null}
            {m.imdb_id ? (
              <a className="inline-flex items-center gap-1 text-muted hover:text-ink" href={`https://www.imdb.com/title/${m.imdb_id}/`} target="_blank" rel="noreferrer">
                IMDb <ExternalLink size={14} strokeWidth={1.5} />
              </a>
            ) : null}
            {m.poster_url?.startsWith("/posters/") || m.overview_source === "wikipedia" ? (
              <span className="text-muted">{t("movie.source")}</span>
            ) : null}
          </div>
        </div>
        </div>
      </section>
      <Similar movieId={m.movie_id} />
    </div>
  );
}

function formatRuntime(minutes: number) {
  const h = Math.floor(minutes / 60);
  return h ? `${h}h ${minutes % 60}m` : `${minutes}m`;
}

const ROW_SOURCES: Record<string, string> = {
  because_you_liked: "Because you liked",
  hidden_gems: "Hidden gems",
  search: "search",
};

function RatingControl({ movie }: { movie: MovieDetail }) {
  const { t } = useI18n();
  const rate = useRate();
  const unrate = useUnrate();
  const value = rate.isPending ? rate.variables.rating : movie.your_rating;
  return (
    <div>
      <p className="mb-1 text-[13px] text-muted">{value ? t("movie.yourRating") : t("movie.rateIt")}</p>
      <div className="flex items-center gap-3">
        <StarRating
          size={26}
          value={value ?? null}
          onChange={(r) => rate.mutate({ movieId: movie.movie_id, rating: r })}
          onClear={() => unrate.mutate(movie.movie_id)}
        />
        {value ? (
          <button type="button" className="text-sm text-muted hover:text-ink" onClick={() => unrate.mutate(movie.movie_id)}>
            {t("movie.clear")}
          </button>
        ) : null}
      </div>
      {rate.isError ? <p className="mt-1 text-[13px] text-danger">{t("movie.rateError")}</p> : null}
    </div>
  );
}

function WatchlistButton({ movie }: { movie: MovieDetail }) {
  const { t } = useI18n();
  const toggle = useWatchlistToggle();
  const inList = toggle.isPending ? toggle.variables.add : movie.in_watchlist;
  return (
    <Button variant="secondary" onClick={() => toggle.mutate({ movieId: movie.movie_id, add: !inList })} aria-pressed={inList}>
      {inList ? <BookmarkCheck size={16} strokeWidth={1.5} /> : <Bookmark size={16} strokeWidth={1.5} />}
      {inList ? t("movie.onList") : t("movie.addList")}
    </Button>
  );
}

function Similar({ movieId }: { movieId: number }) {
  const { t } = useI18n();
  const [mode, setMode] = useState<"cf" | "content">("cf");
  const similar = useQuery({
    queryKey: ["similar", movieId, mode],
    queryFn: () => api<Recommendation[]>(`/movies/${movieId}/similar`, { query: { mode, k: 16 } }),
  });
  return (
    <div className="pt-4">
      <div>
        <Segmented
          label={t("movie.similarKind")}
          value={mode}
          onChange={setMode}
          options={[
            { value: "cf", label: t("reason.sameViewers") },
            { value: "content", label: t("reason.similarStory") },
          ]}
        />
      </div>
      {similar.isError ? <ErrorState error={similar.error} retry={() => similar.refetch()} /> : null}
      {similar.isPending ? (
        <RowSkeleton />
      ) : similar.data ? (
        <MovieRow
          title={mode === "cf" ? t("movie.alsoLiked") : t("movie.moreLike")}
          subtitle={mode === "cf" ? t("movie.cfSub") : t("movie.contentSub")}
          items={similar.data}
          source={`similar_${mode}`}
          note={(r) => t("movie.similarity", { s: r.score.toFixed(2) })}
        />
      ) : null}
    </div>
  );
}
