"use client";

import { useQuery } from "@tanstack/react-query";
import { ChevronRight, FileUp, MessageSquareQuote, Plus, TextSearch, Users } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Backdrop } from "@/components/backdrop";
import { Top10Row } from "@/components/movie-row";
import { Button, ButtonLink, FloatingInput, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { useHealth, useProfile } from "@/lib/hooks";
import { type Key, useI18n } from "@/lib/i18n";
import { authMode, useSession } from "@/lib/session";
import type { Recommendation } from "@/lib/types";

const REASONS: { title: Key; body: Key; Icon: typeof FileUp }[] = [
  { title: "reason1.title", body: "reason1.body", Icon: FileUp },
  { title: "reason2.title", body: "reason2.body", Icon: MessageSquareQuote },
  { title: "reason3.title", body: "reason3.body", Icon: TextSearch },
  { title: "reason4.title", body: "reason4.body", Icon: Users },
];

/** The landing page: a full-screen hero on the stills wall (headline, offer, email sign-up), closed by a
 *  curved red line; then the models' trending forecast, four reasons to join, the FAQ and a second sign-up. */
export function Landing() {
  const { t } = useI18n();
  return (
    <div>
      <section className="relative isolate -mt-16 ml-[calc(50%-50vw)] w-screen">
        <Backdrop />
        <div className="enter mx-auto flex min-h-[600px] max-w-3xl flex-col items-center justify-center px-4 pb-28 pt-32 text-center sm:min-h-[720px]">
          <h1
            className="text-[32px] font-extrabold leading-10 tracking-tight sm:text-[48px] sm:leading-[56px] lg:text-[56px] lg:leading-[64px]"
            style={{ textWrap: "balance" }}
          >
            {t("landing.title")}
          </h1>
          <p className="mt-4 text-lg font-semibold leading-7 sm:text-xl">{t("landing.free")}</p>
          <EmailCta id="hero" />
        </div>
        <div aria-hidden className="relative -mt-14 h-24 overflow-hidden">
          <div className="absolute left-1/2 top-0 h-[360px] w-[180%] -translate-x-1/2 rounded-[50%] border-t-[3px] border-accent bg-bg shadow-[0_-12px_40px_-8px_rgb(214_40_59/0.45)]" />
        </div>
      </section>

      <div className="mx-auto max-w-6xl">
        <TrendingNow />

        <section aria-labelledby="reasons" className="pt-12">
          <h2 id="reasons" className="text-lg font-bold leading-6 sm:text-2xl sm:leading-8">
            {t("landing.reasons")}
          </h2>
          <ul className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {REASONS.map(({ title, body, Icon }) => (
              <li
                key={title}
                className="relative min-h-[240px] rounded-2xl bg-[linear-gradient(149deg,#2e1519_0%,#1d1214_55%,#141114_100%)] p-6 pb-24"
              >
                <h3 className="text-2xl font-bold leading-8">{t(title)}</h3>
                <p className="mt-4 text-base leading-6 text-ink-2">{t(body)}</p>
                <Icon aria-hidden size={64} strokeWidth={1.25} className="absolute bottom-5 right-5 text-accent" />
              </li>
            ))}
          </ul>
        </section>

        <section id="faq" aria-labelledby="faq-title" className="scroll-mt-24 pt-16">
          <h2 id="faq-title" className="text-lg font-bold leading-6 sm:text-2xl sm:leading-8">
            {t("faq.title")}
          </h2>
          <Faq />
          <div className="mx-auto mt-12 flex max-w-3xl flex-col items-center text-center">
            <EmailCta id="closing" />
          </div>
        </section>
      </div>
    </div>
  );
}

/** Email first, like most sign-ups: the address carries over to the account form. Signed-in visitors get a
 *  button to their picks instead. */
export function EmailCta({ id }: { id: string }) {
  const router = useRouter();
  const { t } = useI18n();
  const { ready, session } = useSession();
  const profile = useProfile();
  const [email, setEmail] = useState("");

  if (!ready) return <div className="mt-6 h-[136px]" />;
  if (session) {
    const href = profile.data?.onboarded || profile.data?.n_ratings ? "/home" : "/onboarding";
    return (
      <div className="mt-6">
        <ButtonLink href={href} size="lg" className="sm:h-14 sm:text-xl">
          {t("cta.picks")} <ChevronRight size={24} strokeWidth={2.5} />
        </ButtonLink>
      </div>
    );
  }
  if (authMode === "demo") {
    return (
      <div className="mt-6">
        <ButtonLink href="/login?next=/onboarding" size="lg" className="sm:h-14 sm:text-xl">
          {t("cta.start")} <ChevronRight size={24} strokeWidth={2.5} />
        </ButtonLink>
      </div>
    );
  }
  return (
    <form
      className="mt-6 w-full"
      onSubmit={(e) => {
        e.preventDefault();
        router.push(`/login?mode=signup&next=/onboarding&email=${encodeURIComponent(email.trim())}`);
      }}
    >
      <p className="text-base leading-6 sm:text-xl sm:leading-7">{t("cta.prompt")}</p>
      <div className="mx-auto mt-4 flex max-w-xl flex-col items-center gap-3 sm:flex-row sm:gap-2">
        <FloatingInput
          id={`${id}-email`}
          label={t("cta.email")}
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full sm:flex-1"
        />
        <Button type="submit" size="lg" className="shrink-0 sm:h-14 sm:text-xl">
          {t("cta.start")} <ChevronRight size={24} strokeWidth={2.5} />
        </Button>
      </div>
    </form>
  );
}

/** The ten films the hybrid predicts most viewers will watch next (the crowd forecast), live from the API. */
export function TrendingNow() {
  const { t } = useI18n();
  const top = useQuery({
    queryKey: ["landing-top10"],
    queryFn: () => api<Recommendation[]>("/movies/trending", { query: { k: 10 } }),
  });
  if (top.isError) return null;
  if (!top.data) {
    return (
      <div className="py-4">
        <Skeleton className="h-8 w-48" />
        <div className="flex gap-4 overflow-hidden py-3">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="aspect-[2/3] w-[120px] shrink-0 sm:w-[148px]" />
          ))}
        </div>
      </div>
    );
  }
  return <Top10Row title={t("landing.trending")} subtitle={t("landing.trendingSub")} items={top.data} headingId="trending" />;
}

export function Faq() {
  const { t } = useI18n();
  const health = useHealth();
  const [open, setOpen] = useState<number | null>(null);
  const hit = health.data?.serving_test_metrics["hit@10"];
  const ndcg = health.data?.serving_test_metrics["ndcg@10"];
  const items: { q: string; a: string[] }[] = [
    { q: t("faq.q1"), a: [t("faq.a1a"), t("faq.a1b")] },
    { q: t("faq.q2"), a: [t("faq.a2a"), t("faq.a2b"), t("faq.a2c")] },
    { q: t("faq.q3"), a: [t("faq.a3a"), t("faq.a3b")] },
    { q: t("faq.q4"), a: [t("faq.a4")] },
    {
      q: t("faq.q5"),
      a: [
        hit !== undefined && ndcg !== undefined
          ? t("faq.a5score", { hit: (hit * 100).toFixed(0), ndcg: ndcg.toFixed(3) })
          : t("faq.a5plain"),
        t("faq.a5b"),
      ],
    },
    {
      q: t("faq.q6"),
      a: [(health.data ? t("faq.a6count", { items: health.data.n_items.toLocaleString() }) : "") + t("faq.a6")],
    },
    { q: t("faq.q7"), a: [t("faq.a7a"), t("faq.a7b")] },
  ];

  return (
    <ul className="mt-4 space-y-2">
      {items.map((item, i) => (
        <li key={i}>
          <h3>
            <button
              type="button"
              aria-expanded={open === i}
              aria-controls={`faq-answer-${i}`}
              onClick={() => setOpen(open === i ? null : i)}
              className="flex w-full items-center justify-between gap-4 bg-surface-2 px-6 py-5 text-left text-lg font-medium leading-7 transition-colors duration-200 hover:bg-surface-3 sm:py-6 sm:text-2xl sm:leading-8"
            >
              {item.q}
              <Plus
                aria-hidden
                size={36}
                strokeWidth={1.25}
                className={`shrink-0 transition-transform duration-200 ${open === i ? "rotate-45" : ""}`}
              />
            </button>
          </h3>
          <div
            id={`faq-answer-${i}`}
            hidden={open !== i}
            className="mt-px space-y-4 bg-surface-2 px-6 py-6 text-lg leading-7 sm:text-xl sm:leading-8"
          >
            {item.a.map((p) => (
              <p key={p}>{p}</p>
            ))}
          </div>
        </li>
      ))}
    </ul>
  );
}
