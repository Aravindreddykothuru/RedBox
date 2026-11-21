import type { Metadata } from "next";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui";
import { AccountView } from "./account-view";

export const metadata: Metadata = { title: "Account" };

export default function AccountPage() {
  return (
    <Suspense fallback={<Skeleton className="mt-10 h-96 w-full max-w-2xl" />}>
      <AccountView />
    </Suspense>
  );
}
