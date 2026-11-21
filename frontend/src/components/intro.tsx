"use client";

import { type CSSProperties, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { BRAND } from "@/lib/brand";

/** The opening, about four seconds on black:
 *  1. red light strands sweep in from both sides and fuse into a big R, with a flash of light,
 *  2. a shine runs across the R,
 *  3. the R shrinks into place as the rest of the name unfolds out of it: RED BOX,
 *  4. a shine runs across the name, then it rushes toward the viewer as the strands fly past,
 *  5. the black lifts and the site rises into place.
 *  Any key, click, tap or scroll skips it. Whether it plays is decided before the first paint by
 *  INTRO_SCRIPT (lib/intro.ts); the motion itself is CSS (globals.css, "Opening animation"). */

const DURATION = 4200; // ms, the length of every intro keyframe animation in globals.css
const SKIP_FADE = 250;
const NAME = BRAND.toUpperCase();
// a no-break space keeps its width inside an inline-block letter
const LETTERS = NAME.split("").map((c) => (c === " " ? " " : c));
const INITIAL = NAME[0];
const TONES = ["#d6283b", "#ff6370", "#ff9aa2", "#a8182b", "#ffe3e6", "#e8404f"];

type StrandStyle = CSSProperties & { "--from": string; "--out": string };

// Park-Miller generator with a fixed seed: integer maths gives the same strands on the server and in
// every browser, so hydration matches.
function makeStrands(n: number): StrandStyle[] {
  let seed = 20261120;
  const next = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  return Array.from({ length: n }, (_, i) => {
    const k = next() - 0.5; // where it settles across the big R: -0.5 (left edge) to 0.5 (right edge)
    const h = 0.7 + next() * 0.7; // height, in R font sizes
    const color = TONES[Math.floor(next() * TONES.length)];
    return {
      // the R is about 0.62 of its font size wide (--intro-r, set on .intro)
      left: `calc(50% + ${(k * 0.62).toFixed(3)} * var(--intro-r))`,
      top: `calc(50% - ${(h / 2).toFixed(3)} * var(--intro-r))`,
      height: `calc(${h.toFixed(3)} * var(--intro-r))`,
      width: `${(1 + next() * 4).toFixed(1)}px`,
      background: `linear-gradient(transparent, ${color} 30%, ${color} 70%, transparent)`,
      boxShadow: `0 0 12px ${color}`,
      animationDelay: `${(next() * 0.2).toFixed(2)}s`,
      // even strands come from the left edge, odd ones from the right; at the end each flies outward
      "--from": `${((i % 2 === 0 ? -1 : 1) * (55 + next() * 20)).toFixed(1)}vw`,
      "--out": `${(k * 180).toFixed(1)}vw`,
    };
  });
}
const STRANDS = makeStrands(30);

const subscribe = () => () => {};
// read again on every render, so a skip fading out ("leaving") still counts as playing
const playing = () => ["play", "leaving"].includes(document.documentElement.dataset.intro ?? "");
// "leaving" keeps the intro on screen for its fade but lets the page's delayed entrance (.enter) run now;
// "done" once it is gone
const setPhase = (phase: "leaving" | "done") => {
  document.documentElement.dataset.intro = phase;
};

export function Intro() {
  // the server always renders the intro (CSS hides it unless <html data-intro="play">); after hydration it
  // is removed straight away when this tab has already seen it
  const play = useSyncExternalStore(subscribe, playing, () => true);
  const [leaving, setLeaving] = useState(false);
  const [done, setDone] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!play) return;
    let live = true;
    const finish = () => {
      if (!live) return;
      setPhase("done");
      setDone(true);
    };
    // The CSS animation starts at the first paint, before React hydrates, so end with it rather than
    // with a timer started now: on a slow load a timer would leave an invisible layer blocking clicks.
    const out = ref.current
      ?.getAnimations()
      .find((a) => a instanceof CSSAnimation && a.animationName === "intro-out");
    const timer = out ? undefined : window.setTimeout(finish, DURATION);
    out?.finished.then(finish, () => {}); // rejected when a skip replaces the animation
    const skip = () => {
      setPhase("leaving");
      setLeaving(true);
    };
    const events = ["keydown", "pointerdown", "wheel", "touchmove"] as const;
    for (const e of events) window.addEventListener(e, skip, { passive: true });
    return () => {
      live = false;
      window.clearTimeout(timer);
      for (const e of events) window.removeEventListener(e, skip);
    };
  }, [play]);

  // skipped: a quick fade, then gone
  useEffect(() => {
    if (!leaving) return;
    const fade = window.setTimeout(() => {
      setPhase("done");
      setDone(true);
    }, SKIP_FADE);
    return () => window.clearTimeout(fade);
  }, [leaving]);

  if (!play || done) return null;
  return (
    <div ref={ref} className={`intro${leaving ? " intro-leaving" : ""}`} data-testid="intro">
      <div aria-hidden className="intro-glow" />
      <div aria-hidden className="absolute inset-0">
        {STRANDS.map((style, i) => (
          <span key={i} className="intro-strand" style={style} />
        ))}
      </div>
      <div aria-hidden className="intro-flare" />
      <div aria-hidden className="intro-r">
        <span className="intro-r-letter text-accent">{INITIAL}</span>
        <span className="intro-r-letter intro-r-shine absolute inset-0">{INITIAL}</span>
      </div>
      <div className="intro-mark" role="img" aria-label={BRAND}>
        <span aria-hidden className="intro-word text-accent">
          {LETTERS.map((l, i) => (
            <span key={i} className="intro-letter" style={{ "--i": i } as CSSProperties}>
              {l}
            </span>
          ))}
        </span>
        <span aria-hidden className="intro-word intro-shine absolute inset-0">
          {NAME}
        </span>
      </div>
      <button
        type="button"
        className="intro-skip"
        onClick={() => {
          setPhase("leaving");
          setLeaving(true);
        }}
      >
        Skip intro
      </button>
    </div>
  );
}
