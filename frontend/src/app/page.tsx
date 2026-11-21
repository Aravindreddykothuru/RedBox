import { Suspense } from "react";
import { Landing } from "./landing-client";

// The landing page lives in a client component so it can follow the chosen language (lib/i18n).
export default function LandingPage() {
  return (
    <Suspense fallback={<div className="min-h-[720px]" />}>
      <Landing />
    </Suspense>
  );
}
