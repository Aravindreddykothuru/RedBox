"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import Link from "next/link";
import { useRef } from "react";
import { logEvent } from "@/lib/api";
import type { Movie, Recommendation } from "@/lib/types";
import { MovieCard, movieHref } from "./movie-card";
import { Poster } from "./poster";
import { Skeleton } from "./ui";

// Rows run edge to edge: the scroller cancels the page gutter, and the arrows fill that gutter on each side.
const bleed = "-mx-4 sm:-mx-8 lg:-mx-12";
const track = "no-scrollbar flex snap-x overflow-x-auto px-4 py-3 scroll-px-4 sm:px-8 sm:scroll-px-8 lg:px-12 lg:scroll-px-12";
const arrow =
  "absolute inset-y-3 z-20 hidden w-8 items-center justify-center bg-black/50 text-ink opacity-0 transition-opacity duration-200 hover:bg-black/70 focus-visible:opacity-100 group-hover/row:opacity-100 sm:flex lg:w-12";

function useScroller<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const scroll = (dir: 1 | -1) => {
    const el = ref.current;
    if (el) el.scrollBy({ left: dir * el.clientWidth * 0.8, behavior: "smooth" });
  };
  return { ref, scroll };
}

function Arrows({ scroll }: { scroll: (dir: 1 | -1) => void }) {
  return (
    <>
      <button type="button" aria-label="Scroll left" onClick={() => scroll(-1)} className={`${arrow} left-0 rounded-r`}>
        <ChevronLeft size={32} strokeWidth={2} />
      </button>
      <button type="button" aria-label="Scroll right" onClick={() => scroll(1)} className={`${arrow} right-0 rounded-l`}>
        <ChevronRight size={32} strokeWidth={2} />
      </button>
    </>
  );
}

export function MovieRow({
  title,
  subtitle,
  items,
  source,
  rateable,
  action,
  note,
}: {
  title: string;
  subtitle?: string;
  items: Recommendation[];
  source?: string;
  rateable?: boolean;
  action?: React.ReactNode;
  /** a per-film line, when every film in the row would otherwise repeat the same reason */
  note?: (r: Recommendation) => string;
}) {
  const { ref, scroll } = useScroller<HTMLDivElement>();
  return (
    <section className="py-4" aria-label={title}>
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
        <div>
          <h2 className="text-lg font-bold leading-6 sm:text-xl sm:leading-7">{title}</h2>
          {subtitle ? <p className="mt-1 text-[13px] text-muted">{subtitle}</p> : null}
        </div>
        {action}
      </div>
      <div className={`group/row relative ${bleed}`}>
        <div ref={ref} className={`${track} gap-2`}>
          {items.map((m, i) => (
            <MovieCard key={m.movie_id} movie={m} source={source} position={i} rateable={rateable} note={note?.(m)} />
          ))}
        </div>
        <Arrows scroll={scroll} />
      </div>
    </section>
  );
}

/** A top-ten row: big outlined rank numbers, each with the poster beside it. */
export function Top10Row({
  title,
  subtitle,
  items,
  source,
  headingId,
}: {
  title: string;
  subtitle?: string;
  items: (Movie | Recommendation)[];
  /** when set, clicks are logged as coming from this row */
  source?: string;
  headingId?: string;
}) {
  const { ref, scroll } = useScroller<HTMLOListElement>();
  return (
    <section className="py-4" aria-label={headingId ? undefined : title} aria-labelledby={headingId}>
      <h2 id={headingId} className="text-lg font-bold leading-6 sm:text-2xl sm:leading-8">
        {title}
      </h2>
      {subtitle ? <p className="mt-1 text-[13px] text-muted">{subtitle}</p> : null}
      <div className={`group/row relative ${bleed}`}>
        <ol ref={ref} className={`${track} gap-4`}>
          {items.slice(0, 10).map((m, i) => (
            <li key={m.movie_id} className="shrink-0 snap-start">
              <Link
                href={movieHref(m, source)}
                onClick={() => source && logEvent(m.movie_id, "click", source, i)}
                aria-label={`${i + 1}. ${m.title}${m.year ? ` (${m.year})` : ""}`}
                className="group relative flex w-[184px] items-end justify-end rounded sm:w-[228px]"
              >
                <span aria-hidden className="rank-number absolute -bottom-2 left-0 text-[120px] sm:text-[160px]">
                  {i + 1}
                </span>
                <div className="relative w-[120px] transition-transform duration-300 ease-out group-hover:scale-[1.04] group-hover:delay-100 sm:w-[148px]">
                  <Poster movie={m} className="shadow-[0_4px_12px_rgb(0_0_0/0.5)]" />
                </div>
              </Link>
            </li>
          ))}
        </ol>
        <Arrows scroll={scroll} />
      </div>
    </section>
  );
}

export function RowSkeleton({ count = 7 }: { count?: number }) {
  return (
    <div className="py-4">
      <Skeleton className="h-6 w-48" />
      <div className="flex gap-2 overflow-hidden py-3">
        {Array.from({ length: count }, (_, i) => (
          <div key={i} className="w-[144px] shrink-0 sm:w-[168px] lg:w-[184px]">
            <Skeleton className="aspect-[2/3] w-full" />
            <Skeleton className="mt-3 h-4 w-3/4" />
          </div>
        ))}
      </div>
    </div>
  );
}
