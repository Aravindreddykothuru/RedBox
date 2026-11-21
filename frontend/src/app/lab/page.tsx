import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { LabView } from "./lab-view";

export const metadata: Metadata = { title: "Model Lab" };

export default function LabPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-96 w-full" />}>
      <LabView />
    </Suspense>
  );
}
