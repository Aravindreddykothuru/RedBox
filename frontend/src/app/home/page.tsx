import type { Metadata } from "next";
import { Suspense } from "react";
import { RowSkeleton } from "@/components/movie-row";
import { HomeView } from "./home-view";

export const metadata: Metadata = { title: "For you" };

export default function HomePage() {
  return (
    <Suspense fallback={<RowSkeleton />}>
      <HomeView />
    </Suspense>
  );
}
