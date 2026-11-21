import { assetUrl } from "@/lib/config";
import { StillsWall } from "./stills-wall";

// Behind every full-bleed hero, with the overlay each place needs so white text stays readable on it:
// - landing and sign-in ("hero"): a slowly drifting wall of film stills (components/stills-wall.tsx);
// - home billboard and film page: one scene image, an empty cinema at night with its screen lit
//   (AI-generated with Google Gemini, cropped).
const OVERLAYS = {
  // landing hero and sign-in: the wall dimmed, darkest behind the centred text and at the top and bottom
  hero: [
    "bg-black/60",
    "bg-[radial-gradient(ellipse_at_center,rgb(0_0_0/0.65)_0%,rgb(0_0_0/0.3)_45%,rgb(0_0_0/0)_75%)]",
    "bg-[linear-gradient(0deg,rgb(0_0_0/0.85)_0%,rgb(0_0_0/0)_40%,rgb(0_0_0/0)_65%,rgb(0_0_0/0.8)_100%)]",
  ],
  // home billboard: text sits on the dark left, the lit screen shows on the right, the bottom fades into the page
  billboard: [
    "bg-[linear-gradient(77deg,rgb(0_0_0/0.85)_0%,rgb(0_0_0/0.55)_45%,rgb(0_0_0/0.1)_80%)]",
    "bg-[linear-gradient(0deg,var(--bg)_0%,rgb(20_20_20/0)_38%)]",
  ],
  // film details: mostly dark, a little of the room showing through
  detail: ["bg-black/75", "bg-[linear-gradient(0deg,var(--bg)_0%,rgb(20_20_20/0)_50%)]"],
} as const;

/** ``image``: the film's own poster, blurred into a wash of its colours (home billboard, film page); the
 *  cinema scene when there is none. */
export function Backdrop({ variant = "hero", image }: { variant?: keyof typeof OVERLAYS; image?: string | null }) {
  return (
    <div aria-hidden className="absolute inset-0 -z-10 overflow-hidden bg-black">
      {variant === "hero" ? (
        <StillsWall />
      ) : image ? (
        // eslint-disable-next-line @next/next/no-img-element -- a small poster, blurred: no need for resizing
        <img src={assetUrl(image)} alt="" className="h-full w-full scale-125 object-cover opacity-60 blur-3xl saturate-150" />
      ) : (
        // eslint-disable-next-line @next/next/no-img-element -- a pre-sized local WebP
        <img
          src="/hero/cinema.webp"
          alt=""
          width={1376}
          height={629}
          fetchPriority="high"
          className="h-full w-full object-cover object-[72%_40%]"
        />
      )}
      {OVERLAYS[variant].map((c) => (
        <div key={c} className={`absolute inset-0 ${c}`} />
      ))}
    </div>
  );
}
