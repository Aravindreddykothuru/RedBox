"use client";

import { usePathname } from "next/navigation";
import { useSession, type Session } from "@/lib/session";
import { ButtonLink, Skeleton } from "./ui";

/** Renders children only for a signed-in user; otherwise a short sign-in prompt. */
export function RequireSession({
  children,
  why = "to get recommendations made for you",
}: {
  children: (session: Session) => React.ReactNode;
  why?: string;
}) {
  const { ready, session } = useSession();
  const pathname = usePathname();
  if (!ready) return <Skeleton className="mt-10 h-40 w-full" />;
  if (!session) {
    return (
      <div className="mx-auto mt-16 max-w-md rounded bg-black/70 px-8 py-12 text-center">
        <h1 className="text-[32px] font-bold leading-10">Sign in first</h1>
        <p className="mt-2 text-ink-2">You need an account {why}</p>
        <ButtonLink href={`/login?next=${encodeURIComponent(pathname)}`} className="mt-6">
          Sign in or create an account
        </ButtonLink>
      </div>
    );
  }
  return <>{children(session)}</>;
}
