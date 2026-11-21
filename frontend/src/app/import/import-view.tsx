"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ExternalLink, FileUp } from "lucide-react";
import { useState } from "react";
import { Poster } from "@/components/poster";
import { RequireSession } from "@/components/require-session";
import { Button, ButtonLink, EmptyState, ErrorState, PageHeader } from "@/components/ui";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { BRAND } from "@/lib/brand";
import { useHealth } from "@/lib/hooks";
import type { ImportMatch, ImportPreview, ImportResult, ImportSource } from "@/lib/types";

const MAX_BYTES = 10 * 1024 * 1024;

const SOURCES: { key: ImportSource; name: string; brings: string; steps: string[]; href: string; link: string }[] = [
  {
    key: "netflix",
    name: "Netflix",
    brings: "What you watched. TV episodes are left out",
    steps: ["Open Account, pick your profile, then Viewing activity", "At the bottom of the list, select Download all", "Drop NetflixViewingHistory.csv below"],
    href: "https://www.netflix.com/viewingactivity",
    link: "Open Viewing activity",
  },
  {
    key: "imdb",
    name: "IMDb",
    brings: "Your star ratings, matched by IMDb id. A watchlist export goes to My list",
    steps: ["Open Your ratings", "In the menu next to the list, choose Export", "Drop the CSV below"],
    href: "https://www.imdb.com/list/ratings",
    link: "Open Your ratings",
  },
  {
    key: "letterboxd",
    name: "Letterboxd",
    brings: "Ratings, diary, watched films and watchlist",
    steps: ["Open Settings, then Data", "Choose Export your data", "Drop the .zip below as it is"],
    href: "https://letterboxd.com/settings/data/",
    link: "Open Letterboxd data",
  },
];
const SOURCE_NAMES = Object.fromEntries(SOURCES.map((s) => [s.key, s.name])) as Record<ImportSource, string>;

type Upload = { name: string; content: string; encoding: "text" | "base64" };

/** CSVs go as text; a zip (Letterboxd) as base64, unpacked by the API. */
async function readFile(file: File): Promise<Upload> {
  if (file.size > MAX_BYTES) throw new Error("The file is larger than 10 MB");
  if (/\.zip$/i.test(file.name)) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return { name: file.name, content: btoa(binary), encoding: "base64" };
  }
  return { name: file.name, content: await file.text(), encoding: "text" };
}

const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

export function ImportView() {
  return <RequireSession why="to import your watch history">{() => <Importer />}</RequireSession>;
}

function Importer() {
  const { t } = useI18n();
  const qc = useQueryClient();
  const [fileName, setFileName] = useState<string | null>(null);
  const [readError, setReadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const preview = useMutation({
    mutationFn: (file: Upload) => api<ImportPreview>("/me/import/preview", { method: "POST", body: file }),
    // films already rated or imported start unticked
    onSuccess: (p) => setSelected(new Set(p.items.filter((i) => !i.already).map((i) => i.movie.movie_id))),
  });
  const save = useMutation({
    mutationFn: (p: ImportPreview) =>
      api<ImportResult>("/me/import", {
        method: "POST",
        body: {
          source: p.source,
          items: p.items
            .filter((i) => selected.has(i.movie.movie_id))
            .map((i) => ({ movie_id: i.movie.movie_id, kind: i.kind, rating: i.rating, at: i.at, times: i.times })),
        },
      }),
    onSuccess: () => {
      for (const key of ["recs", "feed", "lab", "profile", "my-ratings", "movie", "seeds", "watchlist", "watched", "new-films"]) {
        qc.invalidateQueries({ queryKey: [key] });
      }
    },
  });

  async function choose(file: File | undefined) {
    if (!file) return;
    setReadError(null);
    setFileName(file.name);
    save.reset();
    try {
      preview.mutate(await readFile(file));
    } catch (e) {
      setReadError(e instanceof Error ? e.message : "The file could not be read");
    }
  }

  if (save.isSuccess) return <Done result={save.data} />;

  return (
    <div className="max-w-4xl">
      <PageHeader title={t("imp.title")}>{t("imp.body")}</PageHeader>
      {preview.data ? (
        <Matches
          data={preview.data}
          fileName={fileName}
          selected={selected}
          setSelected={setSelected}
          saving={save.isPending}
          saveError={save.error}
          onSave={() => save.mutate(preview.data)}
          onReset={() => {
            preview.reset();
            setFileName(null);
          }}
        />
      ) : (
        <>
          <ol className="grid gap-4 md:grid-cols-3">
            {SOURCES.map((s) => (
              <li key={s.key} className="flex flex-col rounded-lg bg-surface p-5">
                <h2 className="text-lg font-bold">{s.name}</h2>
                <p className="mt-1 text-sm text-ink-2">{s.brings}</p>
                <ol className="mt-4 flex-1 list-decimal space-y-1 pl-5 text-sm">
                  {s.steps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
                <a
                  href={s.href}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-4 inline-flex items-center gap-1 self-start rounded text-sm font-semibold text-accent-text hover:underline"
                >
                  {s.link} <ExternalLink size={14} strokeWidth={2} aria-hidden />
                </a>
              </li>
            ))}
          </ol>
          <DropZone onFile={choose} busy={preview.isPending} fileName={fileName} />
          {readError ? <p className="mt-4 text-sm text-danger">{readError}</p> : null}
          {preview.isError ? (
            <div className="mt-4">
              <ErrorState error={preview.error} />
            </div>
          ) : null}
          <p className="mt-6 max-w-2xl text-sm text-muted">
            Your file is read once to find the films and is not kept. Only the films you then choose are saved, and you can
            remove them from your library at any time.
          </p>
        </>
      )}
    </div>
  );
}

function DropZone({ onFile, busy, fileName }: { onFile: (f: File | undefined) => void; busy: boolean; fileName: string | null }) {
  const [over, setOver] = useState(false);
  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        onFile(e.dataTransfer.files[0]);
      }}
      className={`mt-8 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed px-6 py-12 text-center transition-colors duration-150 focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-ink ${
        over ? "border-accent bg-accent/10" : "border-line-strong bg-surface hover:border-ink-2"
      }`}
    >
      <FileUp size={36} strokeWidth={1.5} className="text-ink-2" aria-hidden />
      <span className="text-lg font-semibold">{busy ? `Reading ${fileName ?? "your history"}…` : "Drop your file here, or click to choose it"}</span>
      <span className="text-sm text-muted">NetflixViewingHistory.csv, an IMDb CSV or the Letterboxd .zip, up to 10 MB</span>
      <input
        type="file"
        accept=".csv,.zip,text/csv,application/zip"
        aria-label="Watch history file"
        className="sr-only"
        disabled={busy}
        onChange={(e) => {
          onFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </label>
  );
}

function Matches({
  data,
  fileName,
  selected,
  setSelected,
  saving,
  saveError,
  onSave,
  onReset,
}: {
  data: ImportPreview;
  fileName: string | null;
  selected: Set<number>;
  setSelected: (s: Set<number>) => void;
  saving: boolean;
  saveError: unknown;
  onSave: () => void;
  onReset: () => void;
}) {
  const health = useHealth();
  const ids = data.items.map((i) => i.movie.movie_id);
  const all = ids.length > 0 && ids.every((id) => selected.has(id));
  const skipped = [
    data.tv_skipped ? `${plural(data.tv_skipped, "TV episode")} left out` : null,
    data.short_skipped ? `${plural(data.short_skipped, "brief view")} left out` : null,
  ].filter(Boolean);
  const toggle = (id: number) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  return (
    <section aria-labelledby="matches-title">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 id="matches-title" className="text-2xl font-bold">
            {plural(data.items.length, "film")} found in your {SOURCE_NAMES[data.source]} history
          </h2>
          <p className="mt-1 text-sm text-muted">
            {[fileName, `${plural(data.rows, "line")} read`, ...skipped].filter(Boolean).join(" · ")}
          </p>
        </div>
        <Button variant="ghost" onClick={onReset}>
          Choose another file
        </Button>
      </div>

      {data.unmatched_count ? (
        <details className="mt-4 rounded-lg bg-surface px-4 py-3 text-sm">
          <summary className="cursor-pointer text-ink-2">{plural(data.unmatched_count, "title")} not in the catalogue</summary>
          <p className="mt-2 text-muted">
            {data.unmatched_sample.join(" · ")}
            {data.unmatched_count > data.unmatched_sample.length ? " …" : ""}
          </p>
          <p className="mt-2 text-muted">
            The catalogue holds {health.data ? health.data.n_items.toLocaleString() : "about 8,500"} films that MovieLens viewers
            rated often enough to learn from.
          </p>
        </details>
      ) : null}

      {data.items.length === 0 ? (
        <div className="mt-6">
          <EmptyState title="No films matched">
            Check that this is the export file itself, or try another service.
          </EmptyState>
        </div>
      ) : (
        <>
          <label className="mt-6 flex w-fit cursor-pointer items-center gap-3 text-sm text-ink-2">
            <input
              type="checkbox"
              checked={all}
              onChange={() => setSelected(all ? new Set() : new Set(ids))}
              className="size-4 accent-[var(--accent)]"
            />
            Select all
          </label>
          <ul className="mt-3 divide-y divide-line rounded-lg border border-line bg-surface">
            {data.items.map((item) => (
              <MatchRow key={item.movie.movie_id} item={item} checked={selected.has(item.movie.movie_id)} onToggle={() => toggle(item.movie.movie_id)} />
            ))}
          </ul>
          <div className="sticky bottom-0 mt-6 flex flex-wrap items-center gap-4 border-t border-line bg-bg/95 py-4 backdrop-blur">
            <Button size="lg" disabled={selected.size === 0 || saving} onClick={onSave}>
              {saving ? "Adding…" : `Add ${plural(selected.size, "film")}`}
            </Button>
            <span className="text-sm text-muted">Untick anything you did not enjoy or that matched the wrong film</span>
          </div>
          {saveError ? <ErrorState error={saveError} /> : null}
        </>
      )}
    </section>
  );
}

function MatchRow({ item, checked, onToggle }: { item: ImportMatch; checked: boolean; onToggle: () => void }) {
  const m = item.movie;
  const id = `pick-${m.movie_id}`;
  const what =
    item.kind === "rating"
      ? `Rated ${item.rating} ★`
      : item.kind === "watched"
        ? item.times > 1
          ? `Watched ${item.times} times`
          : "Watched"
        : "For My list";
  const renamed = item.source_title.trim().toLowerCase() !== m.title.trim().toLowerCase();
  return (
    <li className="flex items-center gap-4 px-4 py-3">
      <input id={id} type="checkbox" checked={checked} onChange={onToggle} className="size-4 shrink-0 accent-[var(--accent)]" />
      <div className="w-10 shrink-0">
        <Poster movie={m} sizes="thumb" />
      </div>
      <label htmlFor={id} className="min-w-0 flex-1 cursor-pointer">
        <span className="line-clamp-1 font-medium">
          {m.title}
          {m.year ? <span className="font-normal text-muted"> ({m.year})</span> : null}
        </span>
        <span className="block text-sm text-muted">
          {[what, item.at ? new Date(item.at).toLocaleDateString() : null, renamed ? `listed as “${item.source_title}”` : null]
            .filter(Boolean)
            .join(" · ")}
        </span>
      </label>
      {item.already ? <span className="shrink-0 text-xs text-muted">Already in {BRAND}</span> : null}
    </li>
  );
}

function Done({ result }: { result: ImportResult }) {
  const added = [
    result.watched ? plural(result.watched, "watched film") : null,
    result.ratings ? plural(result.ratings, "rating") : null,
    result.watchlist ? `${plural(result.watchlist, "film")} for My list` : null,
  ].filter(Boolean);
  return (
    <div className="max-w-2xl">
      <PageHeader title="Your history is in">
        {added.length ? `Added ${added.join(", ")}.` : "Nothing new to add."}
        {result.unchanged ? ` ${plural(result.unchanged, "film was", "films were")} already in ${BRAND} and stayed as they were.` : ""}{" "}
        Your picks already reflect it.
      </PageHeader>
      <div className="flex flex-wrap gap-3">
        <ButtonLink href="/home" size="lg">
          See your picks
        </ButtonLink>
        <ButtonLink href="/library?tab=imported" variant="secondary" size="lg">
          Review imported films
        </ButtonLink>
      </div>
    </div>
  );
}
