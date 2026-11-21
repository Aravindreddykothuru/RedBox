"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { RequireSession } from "@/components/require-session";
import { Button, EmptyState, ErrorState, PageHeader, Segmented, Skeleton } from "@/components/ui";
import { api, ApiError } from "@/lib/api";

type Rates = { impressions: number; clicks: number; ctr: number | null };
type Metrics = {
  window_days: number;
  experiment: {
    name: string;
    groups: Record<string, number>;
    by_group: Record<string, Rates>;
    test: {
      control: string;
      treatment: string;
      difference: number | null;
      ci95: [number, number] | null;
      p_value: number | null;
    } | null;
  };
  by_source: Record<string, Rates>;
  activity: { accounts: number; active_users_7d: number; ratings_7d: number };
  catalogue: { items: number; shown: number; share_shown: number };
};

const SOURCE_NAMES: Record<string, string> = {
  hybrid: "Top picks (hybrid)",
  because_you_liked: "Because you liked…",
  hidden_gems: "Hidden gems",
  trending: "Popular right now",
  search: "Search results",
  similar_cf: "Similar: liked by the same viewers",
  similar_content: "Similar: story and themes",
};

const pct = (v: number | null | undefined, digits = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`);

export function OpsView() {
  return <RequireSession why="to see live metrics">{() => <Ops />}</RequireSession>;
}

function Ops() {
  const [days, setDays] = useState<"7" | "30" | "90">("30");
  const metrics = useQuery({
    queryKey: ["ops-metrics", days],
    queryFn: () => api<Metrics>("/ops/metrics", { query: { days } }),
    refetchInterval: 60_000,
  });

  if (metrics.error instanceof ApiError && metrics.error.status === 403) {
    return (
      <div className="mt-10">
        <EmptyState title="Operators only">Live metrics are visible to the accounts listed in the API’s REDBOX_ADMIN_EMAILS.</EmptyState>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="Live metrics">
        How real users respond to the recommendations: click-through rate per row and per A/B-test group. This is
        the online counterpart of the offline Results page.
      </PageHeader>
      <Segmented
        label="Time window"
        value={days}
        onChange={setDays}
        options={[
          { value: "7", label: "7 days" },
          { value: "30", label: "30 days" },
          { value: "90", label: "90 days" },
        ]}
      />
      {metrics.isError ? (
        <div className="mt-6">
          <ErrorState error={metrics.error} retry={() => metrics.refetch()} />
        </div>
      ) : null}
      {!metrics.data ? (
        <Skeleton className="mt-8 h-96 w-full" />
      ) : (
        <Body m={metrics.data} />
      )}
    </div>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface p-5">
      <p className="text-sm text-muted">{label}</p>
      <p className="mt-2 text-[28px] font-medium leading-8 tabular-nums">{value}</p>
    </div>
  );
}

function Body({ m }: { m: Metrics }) {
  const sources = Object.entries(m.by_source).sort((a, b) => b[1].impressions - a[1].impressions);
  const groups = Object.entries(m.experiment.by_group).sort((a, b) => a[0].localeCompare(b[0]));
  const t = m.experiment.test;
  const enough = t && t.p_value !== null;
  const verdict = !enough
    ? "Not enough data yet: both groups need impressions."
    : t!.p_value! < 0.05
      ? `Significant at the 5% level: “${t!.treatment}” ${t!.difference! > 0 ? "raises" : "lowers"} the click-through rate.`
      : "No significant difference yet (p ≥ 0.05). Keep the test running.";

  return (
    <>
      <section className="mt-8 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
        <Tile label="Accounts" value={m.activity.accounts.toLocaleString()} />
        <Tile label="Active users, last 7 days" value={m.activity.active_users_7d.toLocaleString()} />
        <Tile label="Ratings, last 7 days" value={m.activity.ratings_7d.toLocaleString()} />
        <Tile label={`Catalogue shown (${m.window_days} days)`} value={`${pct(m.catalogue.share_shown)} of ${m.catalogue.items.toLocaleString()}`} />
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-medium leading-6">A/B test: {m.experiment.name || "off"}</h2>
        <p className="mt-1 text-sm text-ink-2">
          Each user is assigned to a group by a stable hash of their id. Groups differ in the diversity of their top picks:{" "}
          {Object.entries(m.experiment.groups)
            .map(([g, d]) => `${g} (MMR diversity ${d})`)
            .join(", ")}
          . Users who set the slider themselves are counted as “manual” and left out of the test.
        </p>
        <Table
          head={["Group", "Impressions", "Clicks", "Click-through rate"]}
          rows={groups.map(([g, r]) => [g, r.impressions.toLocaleString(), r.clicks.toLocaleString(), pct(r.ctr, 2)])}
        />
        <div className="mt-4 rounded-xl border border-line bg-surface p-5 text-sm">
          <p className="font-medium">{verdict}</p>
          {enough ? (
            <p className="mt-1 text-ink-2">
              Difference ({t!.treatment} − {t!.control}): {pct(t!.difference, 2)} · 95% interval {pct(t!.ci95![0], 2)} to{" "}
              {pct(t!.ci95![1], 2)} · p = {t!.p_value!.toPrecision(2)} (two-proportion z-test)
            </p>
          ) : null}
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-medium leading-6">Retraining</h2>
        <p className="mt-1 max-w-2xl text-sm text-ink-2">
          The models learn from your users when you retrain them with the app&apos;s ratings. The file uses
          pseudonymous user keys: no emails or account ids. Then run{" "}
          <code className="font-mono text-ink">recsys import-app-ratings ratings.csv</code> and{" "}
          <code className="font-mono text-ink">bash ml/scripts/train_app_models.sh --refresh</code>.
        </p>
        <Button
          variant="secondary"
          className="mt-3"
          onClick={async () => {
            const csv = await api<string>("/ops/ratings.csv", { text: true });
            const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
            const a = document.createElement("a");
            a.href = url;
            a.download = "ratings.csv";
            a.click();
            URL.revokeObjectURL(url);
          }}
        >
          Download ratings for retraining
        </Button>
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-medium leading-6">Click-through rate by row</h2>
        {sources.length ? (
          <Table
            head={["Row or model", "Impressions", "Clicks", "Click-through rate"]}
            rows={sources.map(([s, r]) => [SOURCE_NAMES[s] ?? s, r.impressions.toLocaleString(), r.clicks.toLocaleString(), pct(r.ctr, 2)])}
          />
        ) : (
          <div className="mt-4">
            <EmptyState title="No activity in this window yet" />
          </div>
        )}
      </section>
    </>
  );
}

function Table({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div className="mt-4 overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-sm">
        <thead className="bg-surface-2 text-left text-muted">
          <tr>
            {head.map((h, i) => (
              <th key={h} scope="col" className={`px-4 py-2 font-normal ${i ? "text-right" : ""}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((r) => (
            <tr key={r[0]}>
              {r.map((c, i) => (
                <td key={i} className={`px-4 py-2 ${i ? "text-right font-mono tabular-nums" : "font-medium"}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
