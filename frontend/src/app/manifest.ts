import type { MetadataRoute } from "next";
import { BRAND } from "@/lib/brand";

// Installable web app: phones and desktop browsers can add Red Box to the home screen (public/sw.js
// handles offline navigation).
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: BRAND,
    short_name: BRAND,
    description: "Film picks that tell you why, from a hybrid of collaborative-filtering models.",
    start_url: "/?source=app",
    scope: "/",
    display: "standalone",
    orientation: "portrait-primary",
    background_color: "#141414",
    theme_color: "#141414",
    categories: ["entertainment", "movies"],
    icons: [
      { src: "/icons/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
      { src: "/icons/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
    shortcuts: [
      { name: "Your picks", url: "/home", icons: [{ src: "/icons/icon-192.png", sizes: "192x192" }] },
      { name: "Search", url: "/search", icons: [{ src: "/icons/icon-192.png", sizes: "192x192" }] },
      { name: "Watch together", url: "/group", icons: [{ src: "/icons/icon-192.png", sizes: "192x192" }] },
    ],
  };
}
