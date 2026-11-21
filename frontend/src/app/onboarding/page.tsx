import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { OnboardingView } from "./onboarding-view";

export const metadata: Metadata = { title: "Get started" };

export default function OnboardingPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-96 w-full" />}>
      <OnboardingView />
    </Suspense>
  );
}
