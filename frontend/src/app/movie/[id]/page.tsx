import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { MovieView } from "./movie-view";

export const metadata: Metadata = { title: "Movie" };

export default function MoviePage() {
  return (
    <Suspense
      fallback={
        <div className="grid gap-8 pt-10 md:grid-cols-[280px_1fr]">
          <Skeleton className="aspect-[2/3] w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      }
    >
      <MovieView />
    </Suspense>
  );
}
