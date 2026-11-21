"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Shuffle } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { Poster } from "@/components/poster";
import { Button, ButtonLink, EmptyState, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { MODEL_BLURBS, modelName } from "@/lib/models";
import { useSession } from "@/lib/session";
import type { LabCompare, LabUser, Recommendation } from "@/lib/types";

const K = 10;
const SHORT_NAMES: Record<string, string> = {
  hybrid: "Hybrid",
  ease: "EASE",
  sasrec_ce: "SASRec",
  markov: "Markov",
  ials: "iALS",
  mostpop: "Popular",
};

export function LabView() {
  const { session } = useSession();
  const users = useQuery({ queryKey: ["lab-users"], queryFn: () => api<LabUser[]>("/lab/users", { query: { limit: 500 } }) });
  // "me" = the signed-in user's own ratings; a number = a MovieLens viewer from the training data
  const [subject, setSubject] = useState<"me" | number | null>(null);
  const [diversity, setDiversity] = useState(0);
  const [hovered, setHovered] = useState<number | null>(null);
  const chosen = subject ?? (session ? "me" : (users.data?.[0]?.user_id ?? null));

  const lab = useQuery({
    queryKey: ["lab", session?.userId ?? "anonymous", chosen, diversity],
    queryFn: () =>
      api<LabCompare>("/lab/compare", {
        query: { k: K, diversity, dataset_user: chosen === "me" ? null : chosen },
      }),
    enabled: chosen !== null && (chosen !== "me" || !!session),
    placeholderData: keepPreviousData,
  });

  const models = lab.data ? Object.keys(lab.data.models) : [];

  return (
    <div>
      <PageHeader title="Model Lab">
        The same viewer, every algorithm. Each column is one model&apos;s top {K} for the history shown below.
        Hover a film to see which other models picked it.
      </PageHeader>

      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-line bg-surface p-4">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-ink-2">Viewer</span>
          <select
            value={chosen === null ? "" : String(chosen)}
            onChange={(e) => setSubject(e.target.value === "me" ? "me" : Number(e.target.value))}
            className="h-10 min-w-56 rounded-lg border border-line bg-bg px-3 text-ink"
          >
            {session ? <option value="me">You ({session.name})</option> : null}
            {users.data?.map((u) => (
              <option key={u.user_id} value={u.user_id}>
                MovieLens viewer #{u.user_id} · {u.n_liked} liked
              </option>
            ))}
          </select>
        </label>
        <Button
          variant="secondary"
          className={users.data && !users.data.length ? "hidden" : ""}
          disabled={!users.data?.length}
          onClick={() => {
            const list = users.data ?? [];
            if (list.length) setSubject(list[Math.floor(Math.random() * list.length)].user_id);
          }}
        >
          <Shuffle size={16} strokeWidth={1.5} /> Random viewer
        </Button>
        <label className="flex w-64 flex-col gap-1 text-sm">
          <span className="flex justify-between">
            <span className="text-ink-2">Hybrid diversity (MMR)</span>
            <span className="font-mono text-muted">λ = {(1 - diversity).toFixed(1)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.1}
            value={diversity}
            onChange={(e) => setDiversity(Math.round(Number(e.target.value) * 10) / 10)}
          />
        </label>
      </div>

      {lab.isError ? (
        <div className="mt-6">
          <ErrorState error={lab.error} retry={() => lab.refetch()} />
        </div>
      ) : null}
      {chosen === null && users.isSuccess ? (
        <div className="mt-8">
          <EmptyState title="Sign in to use the Model Lab">
            This deployment doesn&apos;t include MovieLens viewers&apos; histories (their licence doesn&apos;t allow
            redistributing them), so the Lab compares the models on your own ratings.
            <div className="mt-4">
              <ButtonLink href="/login?next=/lab">Sign in</ButtonLink>
            </div>
          </EmptyState>
        </div>
      ) : !lab.data ? (
        <Skeleton className="mt-8 h-[520px] w-full" />
      ) : (
        <div className={lab.isPlaceholderData ? "opacity-60 transition-opacity" : "transition-opacity"}>
          <History items={lab.data.history} isYou={chosen === "me"} />
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-6">
            {models.map((m) => (
              <ModelColumn key={m} model={m} items={lab.data.models[m]} hovered={hovered} setHovered={setHovered} />
            ))}
          </div>
          <Overlap data={lab.data.models} models={models} />
        </div>
      )}
    </div>
  );
}

function History({ items, isYou }: { items: LabCompare["history"]; isYou: boolean }) {
  return (
    <section className="mt-8">
      <h2 className="text-sm font-medium text-muted">
        {isYou ? "Your" : "Their"} most recent liked films ({items.length ? `last ${items.length}` : "none yet"}), oldest first
      </h2>
      {items.length ? (
        <div className="no-scrollbar mt-3 flex gap-2 overflow-x-auto pb-1">
          {items.map((m) => (
            <Link key={m.movie_id} href={`/movie/${m.movie_id}`} className="w-16 shrink-0" title={m.title}>
              <Poster movie={m} sizes="xs" />
            </Link>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-sm text-ink-2">
          With no history every model falls back to popular films. Rate a few films first.
        </p>
      )}
    </section>
  );
}

function ModelColumn({
  model,
  items,
  hovered,
  setHovered,
}: {
  model: string;
  items: Recommendation[];
  hovered: number | null;
  setHovered: (id: number | null) => void;
}) {
  const isHybrid = model === "hybrid";
  return (
    <section
      aria-label={modelName(model)}
      className="rounded-xl border border-line bg-surface p-4"
    >
      <h3 className="flex items-baseline justify-between gap-2 font-medium">
        {modelName(model)}
        {isHybrid ? <span className="font-mono text-xs font-normal text-muted">ours</span> : null}
      </h3>
      <p className="mt-1 min-h-10 text-xs text-muted">{MODEL_BLURBS[model] ?? ""}</p>
      <ol className="mt-3 space-y-1">
        {items.map((r, i) => (
          <li key={r.movie_id}>
            <Link
              href={`/movie/${r.movie_id}?why=${encodeURIComponent(r.reason)}&src=${model}`}
              onMouseEnter={() => setHovered(r.movie_id)}
              onMouseLeave={() => setHovered(null)}
              onFocus={() => setHovered(r.movie_id)}
              onBlur={() => setHovered(null)}
              title={r.reason}
              className={`flex gap-2 rounded-md px-2 py-1 text-sm transition-colors ${
                hovered === r.movie_id ? "bg-accent/[0.12] text-ink" : "text-ink-2 hover:bg-surface-2"
              }`}
            >
              <span className="w-5 shrink-0 text-right font-mono text-xs leading-5 text-muted">{i + 1}</span>
              <span className="line-clamp-1">
                {r.title} <span className="text-muted">{r.year ?? ""}</span>
              </span>
            </Link>
          </li>
        ))}
      </ol>
    </section>
  );
}

/** How much do two models agree? Jaccard overlap of their top-k lists, as a heatmap (one blue ramp). */
function Overlap({ data, models }: { data: Record<string, Recommendation[]>; models: string[] }) {
  const matrix = useMemo(() => {
    const sets = Object.fromEntries(models.map((m) => [m, new Set(data[m].map((r) => r.movie_id))]));
    return models.map((a) =>
      models.map((b) => {
        const inter = [...sets[a]].filter((x) => sets[b].has(x)).length;
        const union = new Set([...sets[a], ...sets[b]]).size;
        return union ? inter / union : 0;
      }),
    );
  }, [data, models]);

  return (
    <section className="mt-12" aria-labelledby="overlap-title">
      <h2 id="overlap-title" className="text-lg font-medium leading-6">
        How much do the models agree?
      </h2>
      <p className="mt-1 max-w-2xl text-sm text-ink-2">
        Share of films two lists have in common (Jaccard overlap: films in both ÷ films in either). Brighter cells
        mean more overlap. Agreement changes from viewer to viewer, so try a few.
      </p>
      <div className="mt-6 overflow-x-auto">
        <table className="border-separate border-spacing-0.5 text-sm">
          <thead>
            <tr>
              <th className="sr-only">Model</th>
              {models.map((m) => (
                <th key={m} scope="col" className="w-16 pb-1 text-xs font-normal text-muted">
                  <abbr title={modelName(m)} className="no-underline">
                    {SHORT_NAMES[m] ?? modelName(m)}
                  </abbr>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((a, i) => (
              <tr key={a}>
                <th scope="row" className="whitespace-nowrap pr-3 text-right text-xs font-normal text-muted">
                  {modelName(a)}
                </th>
                {models.map((b, j) => {
                  const v = matrix[i][j];
                  const diagonal = i === j;
                  return (
                    <td
                      key={b}
                      title={`${modelName(a)} vs ${modelName(b)}: ${(v * 100).toFixed(0)}% overlap`}
                      className="h-12 w-16 rounded text-center font-mono text-xs"
                      style={{
                        background: diagonal
                          ? "var(--surface-2)"
                          : `color-mix(in oklab, #3987e5 ${Math.round(12 + v * 88)}%, var(--surface))`,
                        color: diagonal ? "var(--muted)" : "#f6f2ec",
                      }}
                    >
                      {diagonal ? "—" : `${Math.round(v * 100)}%`}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
