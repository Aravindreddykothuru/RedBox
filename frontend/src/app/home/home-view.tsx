"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Check, Info, Plus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Backdrop } from "@/components/backdrop";
import { movieHref } from "@/components/movie-card";
import { MovieRow, RowSkeleton, Top10Row } from "@/components/movie-row";
import { NewFilmsRow } from "@/components/new-films-row";
import { Poster } from "@/components/poster";
import { RequireSession } from "@/components/require-session";
import { Button, ButtonLink, buttonClass, ErrorState } from "@/components/ui";
import { api, logEvent } from "@/lib/api";
import { useProfile, useWatchlistToggle } from "@/lib/hooks";
import { type Key, useI18n } from "@/lib/i18n";
import type { Session } from "@/lib/session";
import type { Feed, MovieDetail, Recommendation } from "@/lib/types";

const DIVERSITY_LABELS: Record<number, string> = { 0.1: "λ 0.9", 0.2: "λ 0.8", 0.3: "λ 0.7", 0.4: "λ 0.6" };

// row key -> [title, subtitle]; rows not listed keep the API's title
const ROWS: Record<string, [Key, Key]> = {
  because_you_liked: ["row.because", "row.becauseSub"],
  hidden_gems: ["row.gems", "row.gemsSub"],
  trending: ["row.top10", "row.top10Sub"],
};

export function HomeView() {
  return <RequireSession>{(s) => <Home session={s} />}</RequireSession>;
}

function Home({ session }: { session: Session }) {
  const { t, genre } = useI18n();
  const profile = useProfile();
  // null = not chosen by the user: the API applies the user's A/B-test group (see /ops)
  const [diversity, setDiversity] = useState<number | null>(null);
  const picks = useQuery({
    queryKey: ["recs", session.userId, diversity],
    queryFn: () => api<Recommendation[]>("/recommendations", { query: { k: 20, diversity } }),
    placeholderData: keepPreviousData,
  });
  const feed = useQuery({ queryKey: ["feed", session.userId], queryFn: () => api<Feed>("/recommendations/feed") });
  const newUser = profile.data && profile.data.n_ratings + (profile.data.n_watched ?? 0) === 0;
  const nRatings = profile.data?.n_ratings ?? 0;
  const nWatched = profile.data?.n_watched ?? 0;
  const genres = profile.data?.favorite_genres ?? [];
  const basis = [
    nRatings ? t(nRatings === 1 ? "basis.rating1" : "basis.ratings", { n: nRatings }) : null,
    nWatched ? t(nWatched === 1 ? "basis.imported1" : "basis.imported", { n: nWatched }) : null,
  ].filter(Boolean);

  return (
    <div>
      <h1 className="sr-only">{t("home.forYou")}</h1>
      {newUser ? <WelcomeBillboard /> : <Billboard pick={picks.data?.[0]} userId={session.userId} />}

      {picks.isError ? <ErrorState error={picks.error} retry={() => picks.refetch()} /> : null}
      {picks.isPending ? (
        <RowSkeleton />
      ) : picks.data && newUser ? (
        // no rating yet: the models' prediction for viewers who like the genres chosen at sign-up
        <MovieRow
          title={t("row.start")}
          subtitle={
            genres.length
              ? t("row.startGenres", { genres: genres.slice(0, 3).map(genre).join(", ") })
              : t("row.startPlain")
          }
          items={picks.data}
          source="hybrid"
          rateable
        />
      ) : picks.data ? (
        <div className={`transition-opacity duration-150 ${picks.isPlaceholderData ? "opacity-60" : ""}`}>
          <MovieRow
            title={t("row.topPicks")}
            subtitle={basis.length ? t("row.topPicksSub", { basis: basis.join(t("basis.and")) }) : t("row.topPicksPlain")}
            items={picks.data}
            source="hybrid"
            rateable
            action={<DiversityControl value={diversity} onChange={setDiversity} />}
          />
        </div>
      ) : null}

      {profile.data ? <NewFilmsRow userId={session.userId} newUser={Boolean(newUser)} /> : null}

      {feed.isError ? <ErrorState error={feed.error} retry={() => feed.refetch()} /> : null}
      {feed.isPending ? (
        <RowSkeleton />
      ) : (
        feed.data?.rows
          .filter((row) => row.key !== "top_picks")
          .map((row) => {
            const [titleKey, subKey] = ROWS[row.key] ?? [];
            const title =
              row.key === "because_you_liked"
                ? t("row.because", { title: row.title.replace(/^Because you liked /, "") })
                : titleKey
                  ? t(titleKey)
                  : row.title;
            const subtitle = subKey ? t(subKey) : undefined;
            return row.key === "trending" ? (
              <Top10Row key={row.key} title={title} subtitle={subtitle} items={row.items} source={row.key} />
            ) : (
              <MovieRow
                key={row.key}
                title={title}
                subtitle={subtitle}
                items={row.items}
                source={row.key}
                note={ROW_NOTES[row.key]}
                rateable
              />
            );
          })
      )}
    </div>
  );
}

/** The top pick, large: title, why it is here, and two actions, over its own poster blurred. */
function Billboard({ pick, userId }: { pick?: Recommendation; userId: string }) {
  const { t, genre, reason } = useI18n();
  return (
    <section aria-label={t("home.topPick")} className="relative isolate -mt-16 mb-4 ml-[calc(50%-50vw)] w-screen">
      <Backdrop variant="billboard" image={pick?.poster_url} />
      <div className="mx-auto flex min-h-[480px] max-w-[1600px] items-end gap-12 px-4 pb-12 pt-28 sm:min-h-[560px] sm:px-8 lg:px-12">
        {pick ? (
          <>
            <div className="enter max-w-2xl flex-1">
              <p className="text-sm font-semibold text-accent-text">{t("home.topPick")}</p>
              <p className="mt-2 text-[40px] font-extrabold leading-[44px] tracking-tight sm:text-[56px] sm:leading-[60px]" style={{ textWrap: "balance" }}>
                {pick.title}
              </p>
              <p className="mt-3 text-base text-ink-2">
                {[pick.year, pick.genres.slice(0, 3).map(genre).join(" · ")].filter(Boolean).join("  ·  ")}
              </p>
              {pick.reason ? <p className="mt-4 text-lg leading-7 sm:text-xl">{reason(pick.reason)}</p> : null}
              <div className="mt-6 flex flex-wrap gap-3">
                <Link
                  href={movieHref(pick, "hybrid")}
                  onClick={() => logEvent(pick.movie_id, "click", "hybrid", 0)}
                  className={buttonClass("light", "lg")}
                >
                  <Info size={22} strokeWidth={2} /> {t("home.moreInfo")}
                </Link>
                <AddToList key={pick.movie_id} movieId={pick.movie_id} userId={userId} />
              </div>
            </div>
            <Link
              href={movieHref(pick, "hybrid")}
              onClick={() => logEvent(pick.movie_id, "click", "hybrid", 0)}
              tabIndex={-1}
              aria-hidden
              className="hidden w-[220px] shrink-0 lg:block xl:w-[260px]"
            >
              <Poster movie={pick} sizes="lg" className="shadow-[0_16px_40px_rgb(0_0_0/0.6)]" />
            </Link>
          </>
        ) : (
          <div className="h-[200px]" />
        )}
      </div>
    </section>
  );
}

function WelcomeBillboard() {
  const { t } = useI18n();
  return (
    <section aria-label={t("home.welcomeTitle")} className="relative isolate -mt-16 mb-4 ml-[calc(50%-50vw)] w-screen">
      <Backdrop variant="billboard" />
      <div className="mx-auto flex min-h-[480px] max-w-[1600px] items-end px-4 pb-12 pt-28 sm:min-h-[560px] sm:px-8 lg:px-12">
        <div className="enter max-w-2xl">
          <p className="text-[40px] font-extrabold leading-[44px] tracking-tight sm:text-[56px] sm:leading-[60px]">{t("home.welcomeTitle")}</p>
          <p className="mt-4 text-lg leading-7 text-ink-2 sm:text-xl">{t("home.welcomeBody")}</p>
          <div className="mt-6 flex flex-wrap gap-3">
            <ButtonLink href="/onboarding" size="lg">
              {t("home.rateSome")}
            </ButtonLink>
            <ButtonLink href="/import" variant="secondary" size="lg">
              {t("home.importFrom")}
            </ButtonLink>
          </div>
        </div>
      </div>
    </section>
  );
}

/** "My list" for the billboard film; reads the film's details (shared with its page) for the current state. */
function AddToList({ movieId, userId }: { movieId: number; userId: string }) {
  const { t } = useI18n();
  const detail = useQuery({
    queryKey: ["movie", movieId, userId],
    queryFn: () => api<MovieDetail>(`/movies/${movieId}`),
  });
  const toggle = useWatchlistToggle();
  const added = toggle.isPending ? toggle.variables.add : !!detail.data?.in_watchlist;
  return (
    <Button
      variant="secondary"
      size="lg"
      aria-pressed={added}
      disabled={!detail.data}
      onClick={() => toggle.mutate({ movieId, add: !added })}
    >
      {added ? <Check size={22} strokeWidth={2} /> : <Plus size={22} strokeWidth={2} />}
      {t("nav.myList")}
    </Button>
  );
}

function DiversityControl({ value, onChange }: { value: number | null; onChange: (v: number) => void }) {
  const { t } = useI18n();
  const label =
    value === null
      ? t("div.auto")
      : value === 0
        ? `λ 1.0 · ${t("div.accurate")}`
        : value === 0.5
          ? `λ 0.5 · ${t("div.varied")}`
          : DIVERSITY_LABELS[value];
  return (
    <label className="flex w-full max-w-xs flex-col gap-1 sm:w-72">
      <span className="flex justify-between">
        <span className="text-[13px] font-medium text-ink-2">{t("div.label")}</span>
        <span className="text-xs leading-5 text-muted">{label}</span>
      </span>
      <input
        type="range"
        min={0}
        max={0.5}
        step={0.1}
        value={value ?? 0}
        onChange={(e) => onChange(Math.round(Number(e.target.value) * 10) / 10)}
        aria-describedby="diversity-help"
      />
      <span id="diversity-help" className="text-xs text-muted">
        {t("div.help")}
      </span>
    </label>
  );
}

// Rows whose films would all repeat one reason get a per-film fact instead.
const ROW_NOTES: Record<string, ((r: Recommendation) => string) | undefined> = {
  because_you_liked: (r) => `similarity ${r.score.toFixed(2)}`,
};
