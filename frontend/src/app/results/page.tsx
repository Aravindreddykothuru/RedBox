import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { ResultsView } from "./results-view";

export const metadata: Metadata = { title: "Results" };

export default function ResultsPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-96 w-full" />}>
      <ResultsView />
    </Suspense>
  );
}
