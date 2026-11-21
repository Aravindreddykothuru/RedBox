import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { SearchView } from "./search-view";

export const metadata: Metadata = { title: "Search" };

export default function SearchPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-12 w-full max-w-2xl" />}>
      <SearchView />
    </Suspense>
  );
}
