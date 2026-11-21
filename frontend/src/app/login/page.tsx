import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { LoginView } from "./login-view";

export const metadata: Metadata = { title: "Sign in" };

export default function LoginPage() {
  return (
    <Suspense fallback={<Skeleton className="mx-auto mt-16 h-80 max-w-md" />}>
      <LoginView />
    </Suspense>
  );
}
