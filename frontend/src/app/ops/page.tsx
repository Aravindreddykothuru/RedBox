import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { OpsView } from "./ops-view";

export const metadata: Metadata = { title: "Operations" };

export default function OpsPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-96 w-full" />}>
      <OpsView />
    </Suspense>
  );
}
