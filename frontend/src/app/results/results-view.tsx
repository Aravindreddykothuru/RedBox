"use client";

import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { ErrorState, PageHeader, Segmented, Skeleton } from "@/components/ui";
import { useHealth } from "@/lib/hooks";
import { DATASETS, METRIC_COLUMNS, parseResults, type DatasetKey, type ResultRow } from "@/lib/results";
import { CoverageScatter, NdcgBars, TrainingTimeBars } from "./charts";
import { BRAND } from "@/lib/brand";

const FIGURES: Partial<Record<DatasetKey, { src: string; caption: string }[]>> = {
  "ml-1m": [
    {
      src: "/figures/ml-1m/results_by_session.png",
      caption:
        "NDCG@10 by how soon the viewer came back. 96% of test viewers continue the same rating session, where order-aware models shine; for viewers returning more than a day later, EASE wins.",
    },
    {
      src: "/figures/ml-1m/results_ndcg_by_activity.png",
      caption: "NDCG@10 by how many films a viewer had rated (light to heavy users).",
    },
    {
      src: "/figures/ml-1m/hybrid_mmr_tradeoff.png",
      caption: "The diversity slider's trade-off: lower λ (more MMR re-ranking) buys genre variety for a little accuracy.",
    },
    {
      src: "/figures/ml-1m/hybrid_feature_importance.png",
      caption: "What the LightGBM ranker relies on (gain importance of its features).",
    },
  ],
  "amazon-videogames": [
    {
      src: "/figures/amazon-videogames/results_by_session.png",
      caption: "On Amazon there is no rating-session effect: the Markov chain falls below EASE.",
    },
  ],
};

export function ResultsView() {
  const [dataset, setDataset] = useState<DatasetKey>("ml-1m");
  const [showTable, setShowTable] = useState(false);
  const table = useQuery({
    queryKey: ["results-csv", dataset],
    queryFn: async () => {
      const res = await fetch(`/results/${dataset}.csv`);
      if (!res.ok) throw new Error(`results for ${dataset} not found`);
      return parseResults(await res.text());
    },
    staleTime: Infinity,
  });
  const health = useHealth();
  const rows = useMemo(() => table.data?.filter((r) => !r.isVariant) ?? [], [table.data]);
  const hybrid = rows.find((r) => r.isHybrid);
  const bestSingle = rows.filter((r) => !r.isHybrid).sort((a, b) => (b.metrics["NDCG@10"] ?? 0) - (a.metrics["NDCG@10"] ?? 0))[0];
  const gain =
    hybrid && bestSingle ? ((hybrid.metrics["NDCG@10"] ?? 0) / (bestSingle.metrics["NDCG@10"] ?? 1) - 1) * 100 : null;

  return (
    <div>
      <PageHeader title="Results">
        Offline evaluation on the held-out test set: each viewer&apos;s last 10% of interactions, which no model saw
        during training or tuning. Every model ranks the full catalogue.
      </PageHeader>

      <div className="flex flex-wrap items-center justify-between gap-4">
        <Segmented
          label="Dataset"
          value={dataset}
          onChange={setDataset}
          options={Object.entries(DATASETS).map(([value, d]) => ({ value: value as DatasetKey, label: d.label }))}
        />
        <p className="text-sm text-muted">{DATASETS[dataset].note}</p>
      </div>

      {table.isError ? (
        <div className="mt-6">
          <ErrorState error={table.error} />
        </div>
      ) : null}
      {!table.data ? (
        <Skeleton className="mt-8 h-96 w-full" />
      ) : (
        <>
          <section className="mt-8 grid gap-px overflow-hidden rounded-xl border border-line bg-line md:grid-cols-[1.4fr_1fr_1fr_1fr]">
            <div className="bg-surface p-6">
              <p className="text-sm text-muted">{BRAND} hybrid vs the best single model ({bestSingle?.model})</p>
              <p className="mt-3 text-[56px] font-medium leading-[56px] tracking-[-0.03em]">{gain !== null ? `+${gain.toFixed(1)}%` : "—"}</p>
              <p className="mt-2 text-sm text-ink-2">NDCG@10 on the test set; significant at {DATASETS[dataset].significance} (Wilcoxon, Holm-corrected)</p>
            </div>
            <Tile label="Hybrid NDCG@10" value={hybrid?.metrics["NDCG@10"]?.toFixed(4)} />
            <Tile label="Hybrid hit rate@10" value={hybrid?.metrics["HR@10"] !== undefined && hybrid?.metrics["HR@10"] !== null ? `${(hybrid.metrics["HR@10"] * 100).toFixed(1)}%` : undefined} />
            <Tile
              label={dataset === "ml-32m" ? "Serving now: NDCG@10 (live API)" : "Catalogue coverage@10"}
              value={
                dataset === "ml-32m"
                  ? health.data?.serving_test_metrics["ndcg@10"]?.toFixed(4)
                  : hybrid?.metrics["Coverage@10"] !== null && hybrid?.metrics["Coverage@10"] !== undefined
                    ? `${(hybrid.metrics["Coverage@10"] * 100).toFixed(0)}%`
                    : undefined
              }
            />
          </section>

          <div className="mt-10 grid gap-10 xl:grid-cols-2">
            <ChartCard title="Ranking accuracy" subtitle="NDCG@10: how high the films a viewer went on to like appear in their top 10.">
              <NdcgBars rows={rows} />
            </ChartCard>
            <ChartCard
              title="Accuracy vs catalogue coverage"
              subtitle="Coverage@10: share of the catalogue that appears in anyone's top 10. Up and right is better."
            >
              <CoverageScatter rows={rows} />
            </ChartCard>
          </div>
          <div className="mt-10">
            <ChartCard
              title="Training time"
              subtitle="Seconds to train on a laptop CPU (Ryzen 5 7520U), log scale. The hybrid's time excludes its generators."
            >
              <TrainingTimeBars rows={rows} />
            </ChartCard>
          </div>

          <section className="mt-10">
            <button type="button" onClick={() => setShowTable((s) => !s)} className="text-sm font-medium text-ink underline underline-offset-4 hover:text-white" aria-expanded={showTable}>
              {showTable ? "Hide" : "Show"} the full table
            </button>
            {showTable ? <ResultsTable rows={table.data} /> : null}
          </section>

          {FIGURES[dataset]?.length ? (
          <section className="mt-14">
            <h2 className="text-lg font-medium leading-6">From the report</h2>
            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              {FIGURES[dataset].map((f) => (
                <figure key={f.src} className="rounded-xl bg-[#f4efe6] p-3">
                  {/* report figures are pre-rendered PNGs */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={f.src} alt={f.caption} loading="lazy" className="w-full rounded" />
                  <figcaption className="px-1 pt-3 text-sm text-[#3d3730]">{f.caption}</figcaption>
                </figure>
              ))}
            </div>
          </section>
          ) : null}
        </>
      )}
    </div>
  );
}

function Tile({ label, value }: { label: string; value?: string }) {
  return (
    <div className="bg-surface p-6">
      <p className="text-sm text-muted">{label}</p>
      <p className="mt-2 text-[28px] font-medium leading-8 tabular-nums">{value ?? "—"}</p>
    </div>
  );
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-line bg-surface p-5">
      <h2 className="font-medium">{title}</h2>
      <p className="mt-1 text-sm text-muted">{subtitle}</p>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function ResultsTable({ rows }: { rows: ResultRow[] }) {
  return (
    <div className="mt-4 overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-sm">
        <thead className="bg-surface-2 text-left text-muted">
          <tr>
            <th scope="col" className="px-4 py-2 font-normal">Model</th>
            <th scope="col" className="px-4 py-2 font-normal">Family</th>
            {METRIC_COLUMNS.map((c) => (
              <th key={c.key} scope="col" className="px-4 py-2 text-right font-normal">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((r) => (
            <tr key={r.model} className={r.isHybrid ? "bg-surface-2" : ""}>
              <th scope="row" className="whitespace-nowrap px-4 py-2 text-left font-medium">
                {r.model}
              </th>
              <td className="whitespace-nowrap px-4 py-2 text-ink-2">{r.family}</td>
              {METRIC_COLUMNS.map((c) => {
                const v = r.metrics[c.key];
                return (
                  <td key={c.key} className="px-4 py-2 text-right font-mono tabular-nums">
                    {v === null || v === undefined ? "—" : v.toFixed(c.digits)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
