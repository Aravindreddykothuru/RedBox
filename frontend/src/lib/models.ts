// Display names for the serving bundle's models and the offline results table.

import { BRAND } from "@/lib/brand";

export const MODEL_NAMES: Record<string, string> = {
  hybrid: `${BRAND} hybrid`,
  ease: "EASE",
  sasrec_ce: "SASRec",
  sasrec: "SASRec",
  markov: "Markov chain",
  ials: "iALS",
  mostpop: "Most popular",
  lightgcn: "LightGCN",
};

export const MODEL_BLURBS: Record<string, string> = {
  hybrid: "Our two-stage system: every model below proposes candidates, LightGBM ranks them.",
  ease: "A linear model that learns which movies predict which. Best single non-sequential model.",
  sasrec_ce: "A Transformer that reads your ratings in order and predicts what comes next.",
  markov: "Looks only at your most recent movie and what usually follows it.",
  ials: "Matrix factorisation: you and every movie become points in a 'taste space'.",
  mostpop: "The baseline: the most-watched movies, the same for everyone.",
};

export function modelName(key: string): string {
  return MODEL_NAMES[key] ?? key;
}
