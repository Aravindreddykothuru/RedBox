import { PosterImage } from "@/components/poster-image";
import { assetUrl } from "@/lib/config";
import type { Movie } from "@/lib/types";

// Every film has a cover: a typographic poster in a muted tone chosen by its first genre, set in bold type
// like a title card. A film with artwork shows its poster over the cover, fading in once it has loaded, so
// the cover is what a viewer sees while it loads and if it cannot. Covers are the colourful part of the
// interface. (Comedy, the most common first genre, gets a cool tone so feeds don't turn into a wall of
// amber that competes with the accent colour.)
const GENRE_TONES: Record<string, string> = {
  Action: "#7a3328",
  Adventure: "#5b5a2c",
  Animation: "#2d6662",
  "Children's": "#4a6f37",
  Comedy: "#3d5d78",
  Crime: "#363b46",
  Documentary: "#535749",
  Drama: "#56303a",
  Fantasy: "#463b66",
  "Film-Noir": "#2b2a29",
  Horror: "#3f2020",
  Musical: "#713b57",
  Mystery: "#2c3d53",
  Romance: "#7f454d",
  "Sci-Fi": "#1f4b59",
  Thriller: "#4a3a2c",
  War: "#4a4731",
  Western: "#7a5530",
};

type Size = "thumb" | "xs" | "sm" | "lg";

export function Poster({
  movie,
  className = "",
  sizes = "sm",
}: {
  movie: Movie;
  className?: string;
  /** thumb: list thumbnails about 48px wide, too small for a title, so the cover shows its first letter */
  sizes?: Size;
}) {
  if (!movie.poster_url) {
    return <Cover movie={movie} sizes={sizes} className={className} />;
  }
  return (
    <div
      role="img"
      aria-label={`Poster of ${movie.title}`}
      className={`relative aspect-[2/3] w-full overflow-hidden rounded ${className}`}
    >
      <Cover movie={movie} sizes={sizes} decorative className="absolute inset-0 h-full" />
      <PosterImage src={assetUrl(movie.poster_url)} priority={sizes === "lg"} />
    </div>
  );
}

function Cover({
  movie,
  sizes,
  className = "",
  decorative = false,
}: {
  movie: Movie;
  sizes: Size;
  className?: string;
  /** under a poster image: the image's wrapper carries the name */
  decorative?: boolean;
}) {
  const tone = GENRE_TONES[movie.genres[0] ?? ""] ?? "#3a342e";
  const big = sizes === "lg";
  const background = { background: `linear-gradient(180deg, ${tone} 0%, color-mix(in oklab, ${tone} 70%, #0b0908) 100%)` };
  const label = (name: string) => (decorative ? { "aria-hidden": true } : { role: "img", "aria-label": name });
  if (sizes === "thumb") {
    const initial = movie.title.replace(/^(the|a|an)\s+/i, "").match(/[\p{L}\p{N}]/u)?.[0] ?? "";
    return (
      <div
        {...label(movie.title)}
        className={`flex aspect-[2/3] w-full items-center justify-center overflow-hidden rounded ring-1 ring-inset ring-white/10 ${className}`}
        style={background}
      >
        <span aria-hidden className="text-lg font-extrabold text-white/85">
          {initial.toUpperCase()}
        </span>
      </div>
    );
  }
  if (sizes === "xs") {
    return (
      <div
        {...label(movie.title)}
        className={`flex aspect-[2/3] w-full items-end overflow-hidden rounded p-2 ring-1 ring-inset ring-white/10 ${className}`}
        style={background}
      >
        <span className="line-clamp-4 text-[11px] font-bold leading-[13px] text-[#f6efe4] [hyphens:auto] [overflow-wrap:anywhere]">
          {movie.title}
        </span>
      </div>
    );
  }
  return (
    <div
      {...label(`${movie.title}${movie.year ? ` (${movie.year})` : ""}`)}
      className={`flex aspect-[2/3] w-full flex-col justify-between overflow-hidden rounded p-3 ring-1 ring-inset ring-white/10 ${className}`}
      style={background}
    >
      <span className={`font-mono tabular-nums text-white/60 ${big ? "text-[13px]" : "text-xs"}`}>{movie.year ?? ""}</span>
      <div>
        <p
          className={`font-extrabold tracking-tight text-white ${
            big ? "text-[30px] leading-[32px]" : movie.title.length > 28 ? "text-[14px] leading-4" : "text-[17px] leading-5"
          }`}
          style={{ textWrap: "balance" }}
        >
          {movie.title}
        </p>
        <p className={`mt-1 text-white/65 ${big ? "text-[13px]" : "text-xs"}`}>{movie.genres.slice(0, 2).join(" · ")}</p>
      </div>
    </div>
  );
}
