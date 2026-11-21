"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { MovieCard, MovieGrid } from "@/components/movie-card";
import { Poster } from "@/components/poster";
import { RequireSession } from "@/components/require-session";
import { StarRating } from "@/components/star-rating";
import { Button, ButtonLink, EmptyState, ErrorState, PageHeader, Segmented, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { useMyRatings, useRate, useUnrate, useWatchlistToggle } from "@/lib/hooks";
import type { Session } from "@/lib/session";
import type { ImportSource, Movie, RatedMovie, WatchedMovie } from "@/lib/types";

type Tab = "ratings" | "watchlist" | "imported";
const TABS: Tab[] = ["ratings", "watchlist", "imported"];

export function LibraryView() {
  return <RequireSession why="to keep ratings and a watchlist">{(s) => <Library session={s} />}</RequireSession>;
}

function Library({ session }: { session: Session }) {
  const { t } = useI18n();
  const asked = useSearchParams().get("tab") as Tab | null; // e.g. /library?tab=imported after an import
  const [tab, setTab] = useState<Tab>(asked && TABS.includes(asked) ? asked : "ratings");
  return (
    <div>
      <PageHeader title={t("lib.title")}>{t("lib.body")}</PageHeader>
      <Segmented
        label="Library section"
        value={tab}
        onChange={setTab}
        options={[
          { value: "ratings", label: t("lib.ratings") },
          { value: "watchlist", label: t("lib.watchlist") },
          { value: "imported", label: t("lib.imported") },
        ]}
      />
      <div className="mt-8">
        {tab === "ratings" ? <Ratings /> : tab === "watchlist" ? <Watchlist session={session} /> : <Imported session={session} />}
      </div>
    </div>
  );
}

const SOURCE_NAMES: Record<ImportSource, string> = { netflix: "Netflix", imdb: "IMDb", letterboxd: "Letterboxd" };

/** Films from an imported watch history: they count as liked unless rated lower. Each can be rated here
 *  (the rating then decides) or removed. */
function Imported({ session }: { session: Session }) {
  const qc = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const list = useQuery({ queryKey: ["watched", session.userId], queryFn: () => api<WatchedMovie[]>("/me/watched") });
  const forget = useMutation({
    mutationFn: (movieId?: number) =>
      api<void>(movieId === undefined ? "/me/watched" : `/me/watched/${movieId}`, { method: "DELETE" }),
    onSuccess: () => {
      setConfirming(false);
      for (const key of ["watched", "recs", "feed", "lab", "profile", "movie", "seeds", "new-films"]) qc.invalidateQueries({ queryKey: [key] });
    },
  });
  if (list.isError) return <ErrorState error={list.error} retry={() => list.refetch()} />;
  if (list.isPending) return <Skeleton className="h-64 w-full" />;
  if (!list.data.length)
    return (
      <EmptyState title="No imported history">
        Bring in what you have watched on Netflix, IMDb or Letterboxd.
        <ButtonLink href="/import" className="mt-4">
          Import history
        </ButtonLink>
      </EmptyState>
    );
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted">{list.data.length} imported · they count as liked unless you rate them below 4 stars</p>
        {confirming ? (
          <div className="flex items-center gap-2 text-sm">
            <span>Remove all {list.data.length}?</span>
            <Button variant="secondary" size="sm" onClick={() => forget.mutate(undefined)} disabled={forget.isPending}>
              Yes, clear
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setConfirming(false)}>
              Cancel
            </Button>
          </div>
        ) : (
          <div className="flex gap-2">
            <ButtonLink href="/import" variant="ghost" size="sm">
              Import more
            </ButtonLink>
            <Button variant="ghost" size="sm" onClick={() => setConfirming(true)}>
              Clear imported history
            </Button>
          </div>
        )}
      </div>
      <ul className="divide-y divide-line rounded-xl border border-line bg-surface">
        {list.data.map((w) => (
          <ImportedRow key={w.movie_id} item={w} onRemove={() => forget.mutate(w.movie_id)} />
        ))}
      </ul>
    </div>
  );
}

function ImportedRow({ item, onRemove }: { item: WatchedMovie; onRemove: () => void }) {
  const rate = useRate();
  const facts = [
    item.year,
    `watched ${new Date(item.watched_at).toLocaleDateString()}`,
    item.times > 1 ? `${item.times} times` : null,
    `from ${SOURCE_NAMES[item.source] ?? item.source}`,
  ];
  return (
    <li className="flex items-center gap-4 px-4 py-3">
      <Link href={`/movie/${item.movie_id}`} className="w-12 shrink-0">
        <Poster movie={item} sizes="thumb" />
      </Link>
      <div className="min-w-0 flex-1">
        <Link href={`/movie/${item.movie_id}`} className="line-clamp-1 font-medium hover:underline">
          {item.title}
        </Link>
        <p className="text-sm text-muted">{facts.filter(Boolean).join(" · ")}</p>
      </div>
      <StarRating
        size={18}
        value={rate.isSuccess ? rate.variables.rating : null}
        label={`Rate ${item.title}`}
        onChange={(r) => rate.mutate({ movieId: item.movie_id, rating: r })}
      />
      <button
        type="button"
        aria-label={`Remove ${item.title} from imported history`}
        onClick={onRemove}
        className="rounded-md p-2 text-muted hover:bg-surface-2 hover:text-danger"
      >
        <Trash2 size={16} strokeWidth={1.5} />
      </button>
    </li>
  );
}

function Ratings() {
  const ratings = useMyRatings();
  if (ratings.isError) return <ErrorState error={ratings.error} retry={() => ratings.refetch()} />;
  if (ratings.isPending) return <Skeleton className="h-64 w-full" />;
  if (!ratings.data.length)
    return (
      <EmptyState title="No ratings yet">
        <ButtonLink href="/onboarding" className="mt-4">
          Rate some films
        </ButtonLink>
      </EmptyState>
    );
  const liked = ratings.data.filter((r) => r.rating >= 4).length;
  return (
    <div>
      <p className="mb-4 text-sm text-muted">
        {ratings.data.length} rated · {liked} counted as liked (4 stars or more), which is what the models learn from
      </p>
      <ul className="divide-y divide-line rounded-xl border border-line bg-surface">
        {ratings.data.map((r) => (
          <RatingRow key={r.movie_id} item={r} />
        ))}
      </ul>
    </div>
  );
}

function RatingRow({ item }: { item: RatedMovie }) {
  const rate = useRate();
  const unrate = useUnrate();
  const value = rate.isPending ? rate.variables.rating : item.rating;
  return (
    <li className="flex items-center gap-4 px-4 py-3">
      <Link href={`/movie/${item.movie_id}`} className="w-12 shrink-0">
        <Poster movie={item} sizes="thumb" />
      </Link>
      <div className="min-w-0 flex-1">
        <Link href={`/movie/${item.movie_id}`} className="line-clamp-1 font-medium hover:underline">
          {item.title}
        </Link>
        <p className="text-sm text-muted">
          {item.year} · rated {new Date(item.rated_at).toLocaleDateString()}
        </p>
      </div>
      <StarRating
        size={18}
        value={value}
        label={`Your rating for ${item.title}`}
        onChange={(r) => rate.mutate({ movieId: item.movie_id, rating: r })}
      />
      <button
        type="button"
        aria-label={`Remove rating for ${item.title}`}
        onClick={() => unrate.mutate(item.movie_id)}
        className="rounded-md p-2 text-muted hover:bg-surface-2 hover:text-danger"
      >
        <Trash2 size={16} strokeWidth={1.5} />
      </button>
    </li>
  );
}

function Watchlist({ session }: { session: Session }) {
  const list = useQuery({ queryKey: ["watchlist", session.userId], queryFn: () => api<Movie[]>("/me/watchlist") });
  const toggle = useWatchlistToggle();
  if (list.isError) return <ErrorState error={list.error} retry={() => list.refetch()} />;
  if (list.isPending) return <Skeleton className="h-64 w-full" />;
  if (!list.data.length)
    return <EmptyState title="Your watchlist is empty">Open any film and choose &ldquo;Add to watchlist&rdquo;.</EmptyState>;
  return (
    <MovieGrid>
      {list.data.map((m) => (
        <div key={m.movie_id}>
          <MovieCard movie={m} width="w-full" />
          <button
            type="button"
            onClick={() => toggle.mutate({ movieId: m.movie_id, add: false })}
            className="mt-1 text-xs text-muted hover:text-danger"
          >
            Remove
          </button>
        </div>
      ))}
    </MovieGrid>
  );
}
