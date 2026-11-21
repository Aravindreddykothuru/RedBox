import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { LibraryView } from "./library-view";

export const metadata: Metadata = { title: "Library" };

export default function LibraryPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-96 w-full" />}>
      <LibraryView />
    </Suspense>
  );
}
