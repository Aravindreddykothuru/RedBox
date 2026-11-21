import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { GroupView } from "./group-view";

export const metadata: Metadata = { title: "Watch together" };

export default function GroupPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-96 w-full" />}>
      <GroupView />
    </Suspense>
  );
}
