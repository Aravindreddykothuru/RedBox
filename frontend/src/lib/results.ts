// Offline experiment tables (docs/results/<dataset>/tuned/test.csv, copied into public/results/).

import { BRAND, HYBRID_RESULT_NAME } from "@/lib/brand";

export type ResultRow = {
  model: string;
  family: string;
  isHybrid: boolean;
  isVariant: boolean; // hybrid + MMR re-ranking variants
  metrics: Record<string, number | null>;
};

// significance: the hybrid vs the best single model, Wilcoxon signed-rank on per-user NDCG@10, Holm-corrected
export const DATASETS = {
  "ml-1m": {
    label: "MovieLens-1M",
    note: "6,038 viewers · 3,458 movies · ratings ≥ 4 count as liked",
    significance: "p < 10⁻²⁶",
  },
  "amazon-videogames": {
    label: "Amazon Video Games",
    note: "42,045 shoppers · 5,771 games · Amazon Reviews 2023",
    significance: "p < 10⁻²⁶",
  },
  "ml-32m": {
    label: "MovieLens-32M (live app)",
    note: "15,000 test viewers · 8,487 films rated 2018–2023 · the models the app serves",
    significance: "p < 10⁻⁸⁶",
  },
} as const;

export type DatasetKey = keyof typeof DATASETS;

function displayName(model: string): string {
  const mmr = /^hybrid_mmr([\d.]+)$/.exec(model);
  if (mmr) return `Hybrid + MMR (λ = ${mmr[1]})`;
  if (model === HYBRID_RESULT_NAME) return `${BRAND} hybrid`;
  if (model === "SASRec") return "SASRec (BCE loss)"; // the paper's loss; "SASRec (CE loss)" keeps its name
  return model;
}

export function parseResults(csv: string): ResultRow[] {
  const [header, ...lines] = csv.trim().split(/\r?\n/);
  const cols = header.split(",");
  return lines.map((line) => {
    const cells = line.split(",");
    const metrics: Record<string, number | null> = {};
    cols.forEach((c, i) => {
      if (c === "model" || c === "family") return;
      const v = cells[i];
      metrics[c] = v === undefined || v === "" || v === "n/a" ? null : Number(v);
    });
    const raw = cells[cols.indexOf("model")];
    return {
      model: displayName(raw),
      family: cells[cols.indexOf("family")] ?? "",
      isHybrid: raw === HYBRID_RESULT_NAME,
      isVariant: raw.startsWith("hybrid_mmr"),
      metrics,
    };
  });
}

export const METRIC_COLUMNS: { key: string; label: string; digits: number }[] = [
  { key: "NDCG@10", label: "NDCG@10", digits: 4 },
  { key: "HR@10", label: "Hit rate@10", digits: 3 },
  { key: "Recall@10", label: "Recall@10", digits: 4 },
  { key: "MRR@10", label: "MRR@10", digits: 4 },
  { key: "Coverage@10", label: "Coverage@10", digits: 3 },
  { key: "Novelty@10", label: "Novelty@10", digits: 2 },
  { key: "fit (s)", label: "Training (s)", digits: 1 },
];
