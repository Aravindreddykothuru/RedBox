import { ButtonLink } from "@/components/ui";

export default function NotFound() {
  return (
    <div className="mx-auto mt-24 max-w-md pt-10 text-center">
      <p className="font-mono text-xs text-muted">404</p>
      <h1 className="mt-2 text-[40px] font-extrabold leading-[48px] tracking-tight">This reel is missing</h1>
      <p className="mt-2 text-ink-2">The page you asked for doesn&apos;t exist</p>
      <ButtonLink href="/" className="mt-6">
        Back to the start
      </ButtonLink>
    </div>
  );
}
