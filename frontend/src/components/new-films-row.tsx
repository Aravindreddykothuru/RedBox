"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { Recommendation } from "@/lib/types";
import { MovieRow } from "./movie-row";
import { Segmented } from "./ui";

type NewFilms = {
  languages: string[];
  language: string | null;
  refreshed_at: string | null;
  items: Recommendation[];
};
const ALL = "all";

/** "2 hours ago", "yesterday", "3 days ago" in the interface's language. */
function ago(iso: string, lang: string): string {
  const minutes = Math.round((Date.parse(iso) - Date.now()) / 60_000);
  const format = new Intl.RelativeTimeFormat(lang, { numeric: "auto" });
  if (Math.abs(minutes) < 60) return format.format(minutes, "minute");
  if (Math.abs(minutes) < 48 * 60) return format.format(Math.round(minutes / 60), "hour");
  return format.format(Math.round(minutes / 1440), "day");
}

/** New releases: films newer than the models' data (English and Hindi since 2023; Telugu, Tamil, Kannada
 *  and Malayalam since 2016), which the API looks for every day. Ranked for this user from each film's
 *  story and from what Red Box viewers think of it; a brand-new user gets the most talked-about ones in the
 *  genres they picked. Rating one here teaches the row (and every viewer like you). All languages, or
 *  one picked with the filter. Hidden when the API has none. */
export function NewFilmsRow({ userId, newUser = false }: { userId: string; newUser?: boolean }) {
  const { t, genre, lang } = useI18n();
  const [language, setLanguage] = useState(ALL);
  const films = useQuery({
    queryKey: ["new-films", userId, language],
    queryFn: () =>
      api<NewFilms>("/recommendations/new-films", {
        query: language === ALL ? { k: 20 } : { k: 20, language },
      }),
    placeholderData: keepPreviousData, // the current films stay while another language loads
    retry: false,
  });
  const data = films.data;
  if (!data?.languages.length || (language === ALL && !data.items.length)) return null;
  const filter =
    data.languages.length > 1 ? (
      <Segmented
        label={t("row.newFilmsLanguage")}
        value={language}
        options={[
          { value: ALL, label: t("row.newFilmsEveryLanguage") },
          ...data.languages.map((l) => ({ value: l, label: genre(l) })),
        ]}
        onChange={setLanguage}
      />
    ) : null;
  // with a single language the row is simply that language's
  const shown = language === ALL && data.languages.length === 1 ? data.languages[0] : language;
  const about = newUser ? t("row.newFilmsSubNew") : t("row.newFilmsSub");
  const checked = data.refreshed_at ? t("row.newFilmsChecked", { when: ago(data.refreshed_at, lang) }) : null;
  return (
    <div
      aria-busy={films.isPlaceholderData}
      className={`transition-opacity duration-200 ${films.isPlaceholderData ? "opacity-60" : ""}`}
    >
      <MovieRow
        title={shown === ALL ? t("row.newFilmsAll") : t("row.newFilms", { language: genre(shown) })}
        subtitle={checked ? `${about} · ${checked}` : about}
        items={data.items}
        source="new_films"
        action={filter}
        rateable
      />
    </div>
  );
}
