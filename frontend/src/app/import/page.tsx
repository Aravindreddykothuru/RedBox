import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { ImportView } from "./import-view";

export const metadata: Metadata = { title: "Import your watch history" };

export default function ImportPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-96 w-full" />}>
      <ImportView />
    </Suspense>
  );
}
