"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Poster } from "@/components/poster";
import { RequireSession } from "@/components/require-session";
import { StarRating } from "@/components/star-rating";
import { Button, Chip, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { Session } from "@/lib/session";
import type { Movie, Recommendation } from "@/lib/types";

const TARGET = 10;
const MINIMUM = 3;

export function OnboardingView() {
  return <RequireSession why="to save your ratings">{(s) => <Steps session={s} />}</RequireSession>;
}

function Steps({ session }: { session: Session }) {
  const [step, setStep] = useState<"genres" | "rate">("genres");
  const [genres, setGenres] = useState<string[]>([]);
  return step === "genres" ? (
    <GenreStep
      selected={genres}
      toggle={(g) => setGenres((cur) => (cur.includes(g) ? cur.filter((x) => x !== g) : [...cur, g]))}
      next={() => setStep("rate")}
    />
  ) : (
    <RateStep session={session} genres={genres} back={() => setStep("genres")} />
  );
}

function GenreStep({
  selected,
  toggle,
  next,
}: {
  selected: string[];
  toggle: (g: string) => void;
  next: () => void;
}) {
  const { t } = useI18n();
  const genres = useQuery({ queryKey: ["genres"], queryFn: () => api<string[]>("/movies/genres") });
  return (
    <div className="max-w-3xl">
      <PageHeader title={t("onb.genresTitle")}>
        {t("onb.genresBody")}{" "}
        <Link href="/import" className="font-semibold text-ink underline underline-offset-4 hover:text-ink-2">
          {t("onb.importInstead")}
        </Link>
      </PageHeader>
      {genres.isError ? <ErrorState error={genres.error} retry={() => genres.refetch()} /> : null}
      <div className="flex flex-wrap gap-2">
        {genres.isPending
          ? Array.from({ length: 12 }, (_, i) => <Skeleton key={i} className="h-8 w-24 rounded-full" />)
          : genres.data?.map((g) => (
              <Chip key={g} selected={selected.includes(g)} onClick={() => toggle(g)}>
                {g}
              </Chip>
            ))}
      </div>
      <div className="mt-10 flex items-center gap-4">
        <Button onClick={next}>{t("onb.next")}</Button>
        <span className="text-sm text-muted">{t("onb.step1")}</span>
      </div>
    </div>
  );
}

function RateStep({ session, genres, back }: { session: Session; genres: string[]; back: () => void }) {
  const { t } = useI18n();
  const router = useRouter();
  const qc = useQueryClient();
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const seeds = useQuery({
    queryKey: ["seeds", session.userId, genres],
    queryFn: () => api<Movie[]>("/onboarding/seeds", { query: { n: 36, genre: genres } }),
    staleTime: Infinity,
  });
  const finish = useMutation({
    mutationFn: () =>
      api<Recommendation[]>("/onboarding", {
        method: "POST",
        body: {
          favorite_genres: genres,
          display_name: session.name,
          ratings: Object.entries(ratings).map(([movie_id, rating]) => ({ movie_id: Number(movie_id), rating })),
        },
      }),
    onSuccess: () => {
      for (const key of ["profile", "feed", "recs", "my-ratings", "new-films"]) qc.invalidateQueries({ queryKey: [key] });
      router.push("/home");
    },
  });
  const count = Object.keys(ratings).length;

  return (
    <div>
      <PageHeader title={t("onb.rateTitle")}>{t("onb.rateBody")}</PageHeader>
      <div className="sticky top-14 z-20 -mx-4 mb-6 flex flex-wrap items-center gap-4 border-b border-line bg-bg px-4 py-3 sm:-mx-8 sm:px-8 md:top-14">
        <div className="min-w-48 flex-1">
          <p className="text-sm">
            <span className="font-medium tabular-nums">{count}</span> rated
            <span className="text-muted"> · {count >= TARGET ? "that is plenty" : `${TARGET} give good picks`}</span>
          </p>
          <div className="mt-2 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-surface-2">
            <div className="h-full origin-left rounded-full bg-accent transition-transform duration-300 ease-out" style={{ transform: `scaleX(${Math.min(1, count / TARGET)})` }} />
          </div>
        </div>
        <Button variant="ghost" onClick={back}>
          Back
        </Button>
        <Button onClick={() => finish.mutate()} disabled={count < MINIMUM || finish.isPending}>
          {finish.isPending ? t("onb.finding") : count < MINIMUM ? t("onb.rateMore", { n: MINIMUM - count }) : t("onb.show")}
        </Button>
      </div>
      {finish.isError ? <ErrorState error={finish.error} /> : null}
      {seeds.isError ? <ErrorState error={seeds.error} retry={() => seeds.refetch()} /> : null}
      <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        {seeds.isPending
          ? Array.from({ length: 12 }, (_, i) => <Skeleton key={i} className="aspect-[2/3] w-full" />)
          : seeds.data?.map((m) => (
              <div key={m.movie_id}>
                <Poster movie={m} className={ratings[m.movie_id] ? "ring-2 ring-accent" : ""} />
                <p className="mt-2 line-clamp-1 text-sm font-medium">{m.title}</p>
                <p className="text-xs text-muted">{m.year}</p>
                <div className="mt-1">
                  <StarRating
                    size={18}
                    label={`Rate ${m.title}`}
                    value={ratings[m.movie_id] ?? null}
                    onChange={(r) => setRatings((cur) => ({ ...cur, [m.movie_id]: r }))}
                    onClear={() =>
                      setRatings((cur) => {
                        const nextRatings = { ...cur };
                        delete nextRatings[m.movie_id];
                        return nextRatings;
                      })
                    }
                  />
                </div>
              </div>
            ))}
      </div>
    </div>
  );
}
