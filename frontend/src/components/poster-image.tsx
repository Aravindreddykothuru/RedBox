"use client";

import { useState } from "react";

/** A poster image laid over the film's typographic cover. It stays transparent until it has loaded, then
 *  fades in; if it never loads (a slow network, a blocked or missing file) the cover simply stays. So a row
 *  of films is never a row of empty boxes, and a failed image never shows a broken-image icon. */
export function PosterImage({ src, priority = false }: { src: string; priority?: boolean }) {
  const [loaded, setLoaded] = useState(false);
  return (
    // Wikipedia's posters arrive already sized for cards (342 px wide); next/image would add nothing here
    // eslint-disable-next-line @next/next/no-img-element
    <img
      // an image that finished before hydration fires no load event: check it when it mounts
      ref={(img) => {
        if (img?.complete && img.naturalWidth > 0) setLoaded(true);
      }}
      src={src}
      alt=""
      loading={priority ? "eager" : "lazy"}
      fetchPriority={priority ? "high" : "auto"}
      decoding="async"
      onLoad={() => setLoaded(true)}
      className={`absolute inset-0 h-full w-full bg-surface object-cover transition-opacity duration-300 motion-reduce:transition-none ${
        loaded ? "opacity-100" : "opacity-0"
      }`}
    />
  );
}
