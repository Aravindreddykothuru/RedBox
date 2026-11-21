"use client";

import Link from "next/link";
import { forwardRef } from "react";
import { ApiError } from "@/lib/api";

type Variant = "primary" | "light" | "secondary" | "ghost";
type Size = "sm" | "md" | "lg" | "xl";

const base =
  "inline-flex items-center justify-center gap-2 rounded font-semibold transition-[background-color,color,border-color,transform] duration-200 ease-out active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50";
const variants: Record<Variant, string> = {
  // the brand call to action: Red Box red with white text
  primary: "bg-accent text-accent-ink hover:bg-accent-hover",
  // a white button for the main action over imagery ("More info"), and a see-through grey one beside it
  light: "bg-ink text-black hover:bg-white/75",
  secondary: "bg-[rgb(109_109_110/0.7)] text-ink hover:bg-[rgb(109_109_110/0.45)]",
  ghost: "text-ink-2 hover:bg-white/10 hover:text-ink",
};
const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-[13px]",
  md: "h-10 px-4 text-sm",
  lg: "h-12 px-6 text-base",
  xl: "h-14 px-6 text-xl",
};

/** The classes of a button, for links that must look like one and do something on click too. */
export const buttonClass = (variant: Variant = "primary", size: Size = "md") => `${base} ${variants[variant]} ${sizes[size]}`;

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size };

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", size = "md", className = "", ...props },
  ref,
) {
  return <button ref={ref} className={`${base} ${variants[variant]} ${sizes[size]} ${className}`} {...props} />;
});

export function ButtonLink({
  href,
  variant = "primary",
  size = "md",
  className = "",
  children,
}: {
  href: string;
  variant?: Variant;
  size?: Size;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <Link href={href} className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}>
      {children}
    </Link>
  );
}

/** A text field whose label sits inside it and moves up once the field is focused or filled. */
export function FloatingInput({
  id,
  label,
  className = "",
  ...props
}: Omit<React.InputHTMLAttributes<HTMLInputElement>, "placeholder"> & { id: string; label: string }) {
  return (
    <div className={`relative ${className}`}>
      <input
        id={id}
        placeholder=" "
        className="peer h-14 w-full rounded border border-[rgb(128_128_128/0.7)] bg-[rgb(22_22_22/0.7)] px-4 pb-1 pt-5 text-base text-ink focus:border-ink"
        {...props}
      />
      <label
        htmlFor={id}
        className="pointer-events-none absolute left-4 top-4 text-base leading-6 text-ink-2 transition-all duration-150 peer-focus:top-2 peer-focus:text-xs peer-focus:leading-4 peer-[:not(:placeholder-shown)]:top-2 peer-[:not(:placeholder-shown)]:text-xs peer-[:not(:placeholder-shown)]:leading-4"
      >
        {label}
      </label>
    </div>
  );
}

/** A toggle tag (genres, examples). Selected = filled white. */
export function Chip({ selected, onClick, children }: { selected?: boolean; onClick?: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onClick}
      className={`h-8 rounded border px-3 text-[13px] font-medium transition-colors duration-200 ease-out ${
        selected ? "border-ink bg-ink text-black" : "border-line-strong text-ink-2 hover:border-ink-2 hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex max-w-full flex-wrap gap-1 rounded bg-surface p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`h-8 rounded px-3 text-[13px] font-medium transition-colors duration-200 ease-out ${
            value === o.value ? "bg-ink text-black" : "text-ink-2 hover:bg-white/10 hover:text-ink"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Loading placeholder: a still surface step (nothing pulses). */
export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden className={`rounded bg-surface ${className}`} />;
}

export function PageHeader({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <header className="pb-8 pt-10">
      <h1 className="text-[32px] font-bold leading-10 tracking-tight sm:text-[40px] sm:leading-[48px]">{title}</h1>
      {children ? <div className="mt-3 max-w-[44rem] text-base leading-6 text-ink-2">{children}</div> : null}
    </header>
  );
}

export function EmptyState({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="rounded border border-line bg-surface/60 px-6 py-12 text-center">
      <p className="text-lg font-semibold">{title}</p>
      {children ? <div className="mt-2 text-ink-2">{children}</div> : null}
    </div>
  );
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  const message = error instanceof ApiError ? error.message : "Something went wrong while loading this. Please try again";
  return (
    <div role="alert" className="rounded border border-danger/30 bg-danger/[0.08] px-5 py-4">
      <p className="text-ink">{message}</p>
      {retry ? (
        <button type="button" onClick={retry} className="mt-2 text-[13px] font-medium text-ink underline underline-offset-4 hover:text-white">
          Try again
        </button>
      ) : null}
    </div>
  );
}
