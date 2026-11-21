"use client";

import Link from "next/link";
import { API_URL } from "@/lib/api";
import { type Key, useI18n } from "@/lib/i18n";

const COLUMNS: { label: Key; href: string; external?: boolean }[][] = [
  [
    { label: "footer.faq", href: "/#faq" },
    { label: "nav.home", href: "/home" },
    { label: "nav.search", href: "/search" },
  ],
  [
    { label: "nav.myList", href: "/library" },
    { label: "nav.together", href: "/group" },
    { label: "nav.import", href: "/import" },
  ],
  [
    { label: "nav.account", href: "/account" },
    { label: "nav.lab", href: "/lab" },
    { label: "nav.results", href: "/results" },
  ],
  [
    { label: "footer.api", href: `${API_URL}/docs`, external: true },
    { label: "footer.data", href: "https://grouplens.org/datasets/movielens/", external: true },
  ],
];

export function Footer() {
  const { t } = useI18n();
  return (
    <footer className="border-t border-line">
      <div className="mx-auto max-w-6xl px-4 py-12 text-muted sm:px-8 lg:px-12">
        <p className="text-base">
          {t("footer.questions")}{" "}
          <Link href="/#faq" className="underline underline-offset-4 hover:text-ink">
            {t("footer.readFaq")}
          </Link>
        </p>
        <ul className="mt-8 grid grid-cols-2 gap-x-6 gap-y-3 text-[13px] sm:grid-cols-4">
          {COLUMNS.map((col, i) => (
            <li key={i}>
              <ul className="space-y-3">
                {col.map((l) => (
                  <li key={l.label}>
                    {l.external ? (
                      <a href={l.href} target="_blank" rel="noreferrer" className="underline-offset-4 hover:underline">
                        {t(l.label)}
                      </a>
                    ) : (
                      <Link href={l.href} className="underline-offset-4 hover:underline">
                        {t(l.label)}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
        <div className="mt-10 space-y-1 text-[13px]">
          <p>{t("footer.tagline")}</p>
          <p>{t("footer.credits")}</p>
        </div>
      </div>
    </footer>
  );
}
