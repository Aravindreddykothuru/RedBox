"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { WhyOut } from "@/lib/types";

const join = (xs: string[]) => (xs.length > 1 ? `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}` : xs[0] ?? "");

/** "Why this pick?": how much each signal pushed this film up or down in the hybrid's ranker for the
 *  signed-in user. The contributions are exact (TreeSHAP from the LightGBM ranker) and add up to its score;
 *  they are grouped by where the signal comes from. Hidden when the film is not among the user's candidates. */
export function WhyBreakdown({ movieId, userId }: { movieId: number; userId: string }) {
  const { t, signal, lang } = useI18n();
  const why = useQuery({
    queryKey: ["why", movieId, userId],
    queryFn: () => api<WhyOut>(`/recommendations/why/${movieId}`),
  });
  const data = why.data;
  if (!data?.available || !data.parts.length) return null;
  const parts = data.parts.slice(0, 6);
  const widest = Math.max(...parts.map((p) => Math.abs(p.share)));
  return (
    <section aria-labelledby="why-title" className="mt-4 max-w-2xl rounded bg-black/50 px-4 py-4">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id="why-title" className="text-[13px] font-semibold text-accent-text">
          {t("why.title")}
        </h2>
        <p className="text-xs text-muted">{t("why.rank", { rank: data.rank ?? "", n: data.candidates ?? "" })}</p>
      </div>
      <ul className="mt-3 space-y-2">
        {parts.map((p) => {
          const up = p.share >= 0;
          return (
            <li key={p.signal} className="grid grid-cols-[minmax(0,1fr)_96px_44px] items-center gap-3 text-[13px] sm:grid-cols-[minmax(0,1fr)_160px_48px]">
              <span className="truncate text-ink-2" title={signal(p.signal)}>
                {signal(p.signal)}
              </span>
              <span className="h-2 overflow-hidden rounded-full bg-white/10" aria-hidden>
                <span
                  className={`block h-full rounded-full ${up ? "bg-accent" : "bg-white/40"}`}
                  style={{ width: `${Math.max(4, (Math.abs(p.share) / widest) * 100)}%` }}
                />
              </span>
              <span className={`text-right tabular-nums ${up ? "text-ink" : "text-muted"}`}>
                {up ? "+" : "−"}
                {Math.round(Math.abs(p.share) * 100)}%
              </span>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-xs leading-5 text-muted">
        {data.proposed_by.length ? t("why.proposed", { models: lang === "en" ? join(data.proposed_by) : data.proposed_by.join(", ") }) : ""}
        {t("why.note")}
      </p>
    </section>
  );
}
