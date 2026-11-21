import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Noto_Sans_Devanagari, Noto_Sans_Telugu } from "next/font/google";
import { Suspense } from "react";
import { Footer } from "@/components/footer";
import { Intro } from "@/components/intro";
import { Nav, NavShell } from "@/components/nav";
import { BRAND } from "@/lib/brand";
import { INTRO_SCRIPT } from "@/lib/intro";
import { ServiceWorker } from "@/lib/pwa";
import { Providers } from "./providers";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
// Telugu and Hindi interface text (lib/i18n): Geist has no glyphs for those scripts. Loaded only when used.
const telugu = Noto_Sans_Telugu({ variable: "--font-telugu", subsets: ["telugu"], weight: ["400", "600", "700", "800"], preload: false });
const devanagari = Noto_Sans_Devanagari({ variable: "--font-devanagari", subsets: ["devanagari"], weight: ["400", "600", "700", "800"], preload: false });

export const metadata: Metadata = {
  title: { default: BRAND, template: `%s · ${BRAND}` },
  description:
    "Movie recommendations from a two-stage hybrid recommender: several collaborative-filtering models, " +
    "a learning-to-rank model and diversity re-ranking.",
  applicationName: BRAND,
  appleWebApp: { capable: true, title: BRAND, statusBarStyle: "black-translucent" },
};

// the browser chrome and an installed app's title bar match the page
export const viewport: Viewport = { themeColor: "#141414", colorScheme: "dark" };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // suppressHydrationWarning: the head script adds data-intro to <html> before React hydrates
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} ${telugu.variable} ${devanagari.variable} h-full antialiased`} suppressHydrationWarning>
      <head>
        {/* decides before the first paint whether the opening animation plays (see lib/intro.ts) */}
        <script dangerouslySetInnerHTML={{ __html: INTRO_SCRIPT }} />
      </head>
      <body className="flex min-h-full flex-col">
        <Intro />
        <ServiceWorker />
        <Providers>
          {/* the header is fixed and see-through at the top of the page; pages with a full-bleed hero pull
              it up under the header with -mt-16. The nav reads the current path, so it has its own Suspense. */}
          <Suspense fallback={<NavShell />}>
            <Nav />
          </Suspense>
          <main className="mx-auto w-full max-w-[1600px] flex-1 px-4 pb-24 pt-16 sm:px-8 lg:px-12">{children}</main>
          <Footer />
        </Providers>
      </body>
    </html>
  );
}
