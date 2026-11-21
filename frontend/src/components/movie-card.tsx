"use client";

import Link from "next/link";
import { logEvent } from "@/lib/api";
import { useRate } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { Movie, Recommendation } from "@/lib/types";
import { Poster } from "./poster";
import { StarRating } from "./star-rating";

function isRec(m: Movie | Recommendation): m is Recommendation {
  return "reason" in m;
}

export function movieHref(movie: Movie | Recommendation, source?: string) {
  const params = new URLSearchParams();
  if (isRec(movie) && movie.reason) params.set("why", movie.reason);
  if (source) params.set("src", source);
  const qs = params.toString();
  return `/movie/${movie.movie_id}${qs ? `?${qs}` : ""}`;
}

/** A poster tile with the short line saying why the film is here. The poster grows a little on hover.
 *  With `rateable`, a star row lets people rate without leaving the page; it stays out of sight until the
 *  tile is hovered or focused (always shown on touch screens, and once rated), and the recommendations
 *  around it refresh as soon as the rating is saved. */
export function MovieCard({
  movie,
  source,
  position,
  rateable,
  showReason = true,
  note,
  width = "w-[144px] sm:w-[168px] lg:w-[184px]",
}: {
  movie: Movie | Recommendation;
  source?: string;
  position?: number;
  rateable?: boolean;
  showReason?: boolean;
  /** the line to show instead of the film's reason (e.g. a similarity score) */
  note?: string;
  width?: string;
}) {
  const rate = useRate();
  const { genre, reason } = useI18n();
  const saved = rate.isSuccess ? rate.variables?.rating : null;
  return (
    <div className={`group shrink-0 snap-start ${width}`}>
      <Link
        href={movieHref(movie, source)}
        onClick={() => source && logEvent(movie.movie_id, "click", source, position)}
        className="block rounded"
      >
        <div className="transition-transform duration-300 ease-out group-hover:scale-[1.04] group-hover:delay-100 group-focus-visible:scale-[1.04]">
          <Poster movie={movie} className="shadow-[0_4px_12px_rgb(0_0_0/0.4)]" />
        </div>
        <p data-film-title className="mt-3 line-clamp-2 text-[13px] font-semibold leading-[18px] text-ink">
          {movie.title}
        </p>
        {movie.poster_url ? (
          // typographic covers already show year and genre
          <p className="text-xs text-muted">
            {movie.year ?? "—"}
            {movie.genres.length ? ` · ${genre(movie.genres[0])}` : ""}
          </p>
        ) : null}
        {note ? (
          <p className="note mt-1 line-clamp-2">{note}</p>
        ) : showReason && isRec(movie) && movie.reason ? (
          <p className="note mt-1 line-clamp-2">{reason(movie.reason)}</p>
        ) : null}
      </Link>
      {rateable ? (
        <div
          className={`mt-2 transition-opacity duration-150 ease-out ${
            saved
              ? "opacity-100"
              : "opacity-0 group-focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100"
          }`}
        >
          <StarRating
            size={14}
            value={saved ?? null}
            disabled={rate.isPending}
            label={`Rate ${movie.title}`}
            onChange={(r) => rate.mutate({ movieId: movie.movie_id, rating: r })}
          />
        </div>
      ) : null}
    </div>
  );
}

export function MovieGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="enter grid grid-cols-2 gap-x-2 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-7">{children}</div>
  );
}
