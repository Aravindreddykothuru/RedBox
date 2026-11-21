// Copy the offline results (tables + report figures) from docs/ into public/, so the Results page can
// show them without the API. Run after `recsys report` / `recsys hybrid`: `npm run sync-results`.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const pub = resolve(root, "frontend", "public");

const files = [
  ["docs/results/ml-1m/tuned/test.csv", "results/ml-1m.csv"],
  ["docs/results/amazon-videogames/tuned/test.csv", "results/amazon-videogames.csv"],
  ["docs/results/ml-32m/serving/test.csv", "results/ml-32m.csv"], // `recsys serving-eval -c ml-32m`
  ["docs/figures/results/ml-1m/results_by_session.png", "figures/ml-1m/results_by_session.png"],
  ["docs/figures/results/ml-1m/results_ndcg_by_activity.png", "figures/ml-1m/results_ndcg_by_activity.png"],
  ["docs/figures/results/ml-1m/hybrid_mmr_tradeoff.png", "figures/ml-1m/hybrid_mmr_tradeoff.png"],
  ["docs/figures/results/ml-1m/hybrid_feature_importance.png", "figures/ml-1m/hybrid_feature_importance.png"],
  ["docs/figures/results/amazon-videogames/results_by_session.png", "figures/amazon-videogames/results_by_session.png"],
];

let copied = 0;
for (const [src, dest] of files) {
  const from = join(root, src);
  if (!existsSync(from)) {
    console.warn(`missing (skipped): ${src}`);
    continue;
  }
  const to = join(pub, dest);
  mkdirSync(dirname(to), { recursive: true });
  copyFileSync(from, to);
  copied += 1;
}
console.log(`synced ${copied}/${files.length} files into frontend/public`);
