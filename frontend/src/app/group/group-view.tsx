"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { Plus, Search, Users, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useId, useState } from "react";
import { movieHref } from "@/components/movie-card";
import { Poster } from "@/components/poster";
import { RequireSession } from "@/components/require-session";
import { Button, EmptyState, ErrorState, PageHeader, Segmented } from "@/components/ui";
import { api } from "@/lib/api";
import { type Key, useI18n } from "@/lib/i18n";
import { useProfile } from "@/lib/hooks";
import type { Movie, MoviePage } from "@/lib/types";

type Strategy = "average" | "least_misery" | "most_pleasure";
type Friend = { key: number; name: string; films: Movie[] };
type GroupPick = { movie: Movie; score: number; fits: number[] };
type GroupResult = { members: string[]; strategy: Strategy; items: GroupPick[] };

const STRATEGIES: { value: Strategy; label: Key; explain: Key }[] = [
  { value: "average", label: "grp.average", explain: "grp.averageHelp" },
  { value: "least_misery", label: "grp.misery", explain: "grp.miseryHelp" },
  { value: "most_pleasure", label: "grp.pleasure", explain: "grp.pleasureHelp" },
];
const MAX_FRIENDS = 5;
const MAX_FILMS = 8;

export function GroupView() {
  return <RequireSession why="to pick films with friends">{() => <WatchTogether />}</RequireSession>;
}

function WatchTogether() {
  const { t } = useI18n();
  const profile = useProfile();
  const [includeMe, setIncludeMe] = useState(true);
  const [friends, setFriends] = useState<Friend[]>([{ key: 1, name: "", films: [] }]);
  const [strategy, setStrategy] = useState<Strategy>("average");
  const find = useMutation({
    mutationFn: () =>
      api<GroupResult>("/group", {
        method: "POST",
        body: {
          include_me: includeMe,
          strategy,
          k: 12,
          friends: friends
            .filter((f) => f.films.length)
            .map((f, i) => ({ name: f.name.trim() || `Friend ${i + 1}`, liked: f.films.map((m) => m.movie_id) })),
        },
      }),
  });
  const ready = friends.filter((f) => f.films.length).length + (includeMe ? 1 : 0) >= 2;
  const mine = (profile.data?.n_ratings ?? 0) + (profile.data?.n_watched ?? 0);
  const update = (key: number, change: Partial<Friend>) =>
    setFriends((all) => all.map((f) => (f.key === key ? { ...f, ...change } : f)));

  return (
    <div className="max-w-5xl">
      <PageHeader title={t("nav.together")}>{t("grp.body")}</PageHeader>

      <section aria-labelledby="who" className="space-y-3">
        <h2 id="who" className="text-lg font-bold">
          {t("grp.who")}
        </h2>
        <label className="flex cursor-pointer items-center gap-3 rounded-lg bg-surface px-4 py-3">
          <input
            type="checkbox"
            checked={includeMe}
            onChange={(e) => setIncludeMe(e.target.checked)}
            className="size-4 accent-[var(--accent)]"
          />
          <span className="font-semibold">{t("grp.you")}</span>
          <span className="text-sm text-muted">
            {mine ? `from your ${mine} rated and imported films` : "you have no ratings yet, so the models use what many viewers like"}
          </span>
        </label>
        {friends.map((f, i) => (
          <FriendCard
            key={f.key}
            index={i}
            friend={f}
            onName={(name) => update(f.key, { name })}
            onAdd={(m) => update(f.key, { films: f.films.some((x) => x.movie_id === m.movie_id) ? f.films : [...f.films, m].slice(0, MAX_FILMS) })}
            onRemoveFilm={(id) => update(f.key, { films: f.films.filter((x) => x.movie_id !== id) })}
            onRemove={friends.length > 1 ? () => setFriends((all) => all.filter((x) => x.key !== f.key)) : undefined}
          />
        ))}
        {friends.length < MAX_FRIENDS ? (
          <Button
            variant="ghost"
            onClick={() => setFriends((all) => [...all, { key: Math.max(...all.map((x) => x.key)) + 1, name: "", films: [] }])}
          >
            <Plus size={18} strokeWidth={2} aria-hidden /> {t("grp.addFriend")}
          </Button>
        ) : null}
      </section>

      <section aria-labelledby="how" className="mt-8">
        <h2 id="how" className="text-lg font-bold">
          {t("grp.how")}
        </h2>
        <div className="mt-3">
          <Segmented label="Group strategy" value={strategy} onChange={setStrategy} options={STRATEGIES.map(({ value, label }) => ({ value, label: t(label) }))} />
        </div>
        <p className="mt-2 max-w-2xl text-sm text-ink-2">{t(STRATEGIES.find((s) => s.value === strategy)?.explain ?? "grp.averageHelp")}</p>
      </section>

      <div className="mt-8 flex flex-wrap items-center gap-4">
        <Button size="lg" disabled={!ready || find.isPending} onClick={() => find.mutate()}>
          <Users size={20} strokeWidth={2} aria-hidden /> {find.isPending ? t("grp.asking") : t("grp.find")}
        </Button>
        {!ready ? <span className="text-sm text-muted">{t("grp.needFriend")}</span> : null}
      </div>

      {find.isError ? (
        <div className="mt-6">
          <ErrorState error={find.error} />
        </div>
      ) : null}
      {find.data ? <Results result={find.data} /> : null}
    </div>
  );
}

function FriendCard({
  index,
  friend,
  onName,
  onAdd,
  onRemoveFilm,
  onRemove,
}: {
  index: number;
  friend: Friend;
  onName: (name: string) => void;
  onAdd: (m: Movie) => void;
  onRemoveFilm: (id: number) => void;
  onRemove?: () => void;
}) {
  const nameId = useId();
  return (
    <div className="rounded-lg bg-surface px-4 py-4">
      <div className="flex flex-wrap items-center gap-3">
        <label htmlFor={nameId} className="sr-only">
          Friend {index + 1}&apos;s name
        </label>
        <input
          id={nameId}
          value={friend.name}
          onChange={(e) => onName(e.target.value)}
          placeholder={`Friend ${index + 1}`}
          maxLength={40}
          className="h-10 w-48 rounded border border-line-strong bg-black/40 px-3 font-semibold text-ink placeholder:font-normal placeholder:text-muted focus:border-ink focus:outline-none"
        />
        <span className="text-sm text-muted">Films they love ({friend.films.length}/{MAX_FILMS})</span>
        {onRemove ? (
          <button type="button" onClick={onRemove} className="ml-auto text-sm text-muted hover:text-ink">
            Remove
          </button>
        ) : null}
      </div>
      <ul className="mt-3 flex flex-wrap gap-2">
        {friend.films.map((m) => (
          <li key={m.movie_id} className="inline-flex h-8 items-center gap-2 rounded bg-surface-2 pl-3 pr-1 text-[13px]">
            {m.title}
            {m.year ? <span className="text-muted">{m.year}</span> : null}
            <button
              type="button"
              aria-label={`Remove ${m.title}`}
              onClick={() => onRemoveFilm(m.movie_id)}
              className="rounded p-1 text-muted hover:text-ink"
            >
              <X size={14} strokeWidth={2} />
            </button>
          </li>
        ))}
      </ul>
      {friend.films.length < MAX_FILMS ? <FilmPicker onPick={onAdd} label={`Add a film ${friend.name || `friend ${index + 1}`} loves`} /> : null}
    </div>
  );
}

/** Title search with a short list of matches to pick from. */
function FilmPicker({ onPick, label }: { onPick: (m: Movie) => void; label: string }) {
  const [text, setText] = useState("");
  const [query, setQuery] = useState("");
  useEffect(() => {
    const t = window.setTimeout(() => setQuery(text.trim()), 250);
    return () => window.clearTimeout(t);
  }, [text]);
  const results = useQuery({
    queryKey: ["film-picker", query],
    queryFn: () => api<MoviePage>("/movies", { query: { q: query, limit: 6 } }),
    enabled: query.length >= 2,
  });
  const listId = useId();
  return (
    <div className="relative mt-3 max-w-md">
      <Search size={16} strokeWidth={2} className="pointer-events-none absolute left-3 top-3 text-muted" aria-hidden />
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        aria-label={label}
        aria-controls={listId}
        placeholder="Search a film…"
        className="h-10 w-full rounded border border-line-strong bg-black/40 pl-9 pr-3 text-sm text-ink placeholder:text-muted focus:border-ink focus:outline-none"
      />
      {query.length >= 2 && results.data?.items.length ? (
        <ul id={listId} className="absolute inset-x-0 top-11 z-10 overflow-hidden rounded border border-line-strong bg-surface-2 shadow-[0_12px_32px_rgb(0_0_0/0.5)]">
          {results.data.items.map((m) => (
            <li key={m.movie_id}>
              <button
                type="button"
                onClick={() => {
                  onPick(m);
                  setText("");
                  setQuery("");
                }}
                className="flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-surface-3"
              >
                {/* decorative: the title next to it names the button */}
                <span className="w-7 shrink-0" aria-hidden>
                  <Poster movie={m} sizes="thumb" />
                </span>
                <span className="min-w-0 flex-1 truncate">{m.title}</span>
                <span className="text-muted">{m.year}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Results({ result }: { result: GroupResult }) {
  if (!result.items.length) {
    return (
      <div className="mt-8">
        <EmptyState title="No films found">Try adding more films for each person.</EmptyState>
      </div>
    );
  }
  return (
    <section aria-labelledby="picks" className="mt-10">
      <h2 id="picks" className="text-2xl font-bold">
        Films for {result.members.length === 2 ? result.members.join(" and ") : `${result.members.slice(0, -1).join(", ")} and ${result.members.at(-1)}`}
      </h2>
      <p className="mt-1 text-sm text-muted">
        Fit: how well each film suits each person, from the hybrid&apos;s scores for every member (100% = their best match
        among these films)
      </p>
      <ol className="mt-4 divide-y divide-line rounded-lg border border-line bg-surface">
        {result.items.map((it, rank) => (
          <li key={it.movie.movie_id} className="flex items-center gap-4 px-4 py-3">
            <span className="w-6 shrink-0 text-right text-lg font-bold tabular-nums text-muted">{rank + 1}</span>
            <Link href={movieHref(it.movie)} className="w-12 shrink-0">
              <Poster movie={it.movie} sizes="thumb" />
            </Link>
            <div className="min-w-0 flex-1">
              <Link href={movieHref(it.movie)} className="line-clamp-1 font-semibold hover:underline">
                {it.movie.title}
              </Link>
              <p className="text-sm text-muted">{[it.movie.year, it.movie.genres.slice(0, 3).join(" · ")].filter(Boolean).join(" · ")}</p>
            </div>
            <ul className="hidden w-[min(360px,40%)] shrink-0 space-y-1 sm:block" aria-label={`Fit for each person: ${it.movie.title}`}>
              {result.members.map((name, i) => (
                <li key={name + i} className="grid grid-cols-[72px_1fr_36px] items-center gap-2 text-xs">
                  <span className="truncate text-ink-2">{name}</span>
                  <span className="h-1.5 overflow-hidden rounded-full bg-white/10" aria-hidden>
                    <span className="block h-full rounded-full bg-accent" style={{ width: `${Math.round(it.fits[i] * 100)}%` }} />
                  </span>
                  <span className="text-right tabular-nums text-ink-2">{Math.round(it.fits[i] * 100)}%</span>
                </li>
              ))}
            </ul>
            <span className="text-sm tabular-nums text-ink-2 sm:hidden">{Math.round(Math.min(...it.fits) * 100)}% for all</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
