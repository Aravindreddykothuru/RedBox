"use client";

import { useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, Languages, Search } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { BRAND } from "@/lib/brand";
import { type Key, LANGS, useI18n } from "@/lib/i18n";
import { useInstallPrompt } from "@/lib/pwa";
import { signOut, useSession } from "@/lib/session";

const LINKS: { href: string; label: Key }[] = [
  { href: "/home", label: "nav.home" },
  { href: "/search", label: "nav.search" },
  { href: "/library", label: "nav.myList" },
  { href: "/group", label: "nav.together" },
  { href: "/lab", label: "nav.lab" },
  { href: "/results", label: "nav.results" },
];

export function Wordmark() {
  return (
    <span className="text-[22px] font-extrabold uppercase leading-none tracking-[-0.04em] text-accent sm:text-[26px]">
      {BRAND}
    </span>
  );
}

// The header is see-through over the top of the page and turns solid once the page scrolls.
function subscribeScroll(onChange: () => void) {
  window.addEventListener("scroll", onChange, { passive: true });
  return () => window.removeEventListener("scroll", onChange);
}
const useScrolled = () =>
  useSyncExternalStore(
    subscribeScroll,
    () => window.scrollY > 8,
    () => false,
  );

const headerClass = (solid: boolean) =>
  `fixed inset-x-0 top-0 z-40 transition-colors duration-300 ${solid ? "bg-bg" : "bg-linear-to-b from-black/80 to-transparent"}`;
const barClass = "mx-auto flex h-16 max-w-[1600px] items-center gap-6 px-4 sm:px-8 lg:gap-10 lg:px-12";

export function Nav() {
  const pathname = usePathname();
  const router = useRouter();
  const qc = useQueryClient();
  const pwa = useInstallPrompt();
  const { t } = useI18n();
  const { ready, session } = useSession();
  const scrolled = useScrolled();
  const links: { href: string; label: Key }[] = session?.isAdmin ? [...LINKS, { href: "/ops", label: "nav.ops" }] : LINKS;
  const isActive = (href: string) => pathname.startsWith(href);

  return (
    <header className={headerClass(scrolled)}>
      <div className={barClass}>
        <Link href={session ? "/home" : "/"} aria-label={`${BRAND}, home`} className="shrink-0 rounded">
          <Wordmark />
        </Link>
        {session ? (
          <>
            <nav aria-label="Main" className="hidden items-center gap-5 xl:flex">
              {links.map((l) => (
                <Link
                  key={l.href}
                  href={l.href}
                  aria-current={isActive(l.href) ? "page" : undefined}
                  className={`whitespace-nowrap text-sm transition-colors duration-200 ${
                    isActive(l.href) ? "font-semibold text-ink" : "text-[#e5e5e5] hover:text-ink-2"
                  }`}
                >
                  {t(l.label)}
                </Link>
              ))}
            </nav>
            <div className="xl:hidden">
              <Menu trigger={<span className="text-sm font-medium">{t("nav.browse")}</span>} align="left">
                {(close) => (
                  <ul className="w-56 py-2">
                    {links.map((l) => (
                      <li key={l.href}>
                        <Link
                          href={l.href}
                          onClick={close}
                          aria-current={isActive(l.href) ? "page" : undefined}
                          className={`block px-4 py-3 text-center text-sm hover:bg-white/10 ${
                            isActive(l.href) ? "font-semibold text-ink" : "text-ink-2"
                          }`}
                        >
                          {t(l.label)}
                        </Link>
                      </li>
                    ))}
                  </ul>
                )}
              </Menu>
            </div>
          </>
        ) : null}

        <div className="ml-auto flex items-center gap-4 sm:gap-5">
          <LanguageMenu />
          {!ready ? null : session ? (
            <>
              <Link href="/search" aria-label={t("nav.search")} className="rounded p-1 text-ink transition-colors duration-200 hover:text-ink-2">
                <Search size={20} strokeWidth={2} />
              </Link>
              <Menu label={t("nav.accountMenu")} trigger={<Avatar name={session.name} />} align="right">
                {(close) => (
                  <div className="w-60 py-2 text-[13px]">
                    <div className="flex items-center gap-3 px-4 py-2">
                      <Avatar name={session.name} />
                      <span className="truncate text-ink">{session.name}</span>
                    </div>
                    <MenuLink href="/account" onClick={close}>
                      {t("nav.account")}
                    </MenuLink>
                    <MenuLink href="/library" onClick={close}>
                      {t("nav.myList")}
                    </MenuLink>
                    <MenuLink href="/import" onClick={close}>
                      {t("nav.import")}
                    </MenuLink>
                    <MenuLink href="/group" onClick={close}>
                      {t("nav.together")}
                    </MenuLink>
                    {session.isAdmin ? (
                      <MenuLink href="/ops" onClick={close}>
                        {t("nav.ops")}
                      </MenuLink>
                    ) : null}
                    {pwa.canInstall ? (
                      <button
                        type="button"
                        onClick={() => {
                          close();
                          void pwa.install();
                        }}
                        className="block w-full px-4 py-2 text-left text-ink-2 hover:text-ink hover:underline"
                      >
                        {t("nav.install")}
                      </button>
                    ) : null}
                    <div className="my-2 border-t border-line-strong" />
                    <button
                      type="button"
                      onClick={async () => {
                        close();
                        await signOut();
                        qc.clear();
                        router.push("/");
                      }}
                      className="block w-full px-4 py-2 text-center text-ink hover:underline"
                    >
                      {t("nav.signOut")}
                    </button>
                  </div>
                )}
              </Menu>
            </>
          ) : pathname.startsWith("/login") ? null : (
            <Link
              href="/login"
              className="inline-flex h-8 items-center whitespace-nowrap rounded bg-accent px-4 text-sm font-semibold text-accent-ink transition-colors duration-200 hover:bg-accent-hover"
            >
              {t("nav.signIn")}
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}

/** English, Telugu or Hindi for the interface; remembered in this browser. */
function LanguageMenu() {
  const { lang, setLang, t } = useI18n();
  const current = LANGS.find((l) => l.code === lang) ?? LANGS[0];
  return (
    <Menu
      label={`${t("nav.language")}: ${current.native}`}
      trigger={
        <span className="flex items-center gap-1.5 text-sm text-ink-2 hover:text-ink">
          <Languages size={18} strokeWidth={2} aria-hidden />
          <span className="font-medium">{current.short}</span>
        </span>
      }
      align="right"
    >
      {(close) => (
        <ul className="w-44 py-2 text-sm" aria-label={t("nav.language")}>
          {LANGS.map((l) => (
            <li key={l.code}>
              <button
                type="button"
                lang={l.code}
                aria-current={l.code === lang ? "true" : undefined}
                onClick={() => {
                  setLang(l.code);
                  close();
                }}
                className="flex w-full items-center justify-between px-4 py-2 text-left text-ink-2 hover:bg-white/10 hover:text-ink"
              >
                {l.native}
                {l.code === lang ? <Check size={16} strokeWidth={2} className="text-accent-text" aria-hidden /> : null}
              </button>
            </li>
          ))}
        </ul>
      )}
    </Menu>
  );
}

/** Static stand-in while the interactive nav loads. */
export function NavShell() {
  return (
    <header className={headerClass(false)}>
      <div className={barClass}>
        <Wordmark />
      </div>
    </header>
  );
}

function Avatar({ name }: { name: string }) {
  return (
    <span aria-hidden className="flex size-8 shrink-0 items-center justify-center rounded bg-accent text-sm font-bold text-accent-ink">
      {(name.trim()[0] ?? "?").toUpperCase()}
    </span>
  );
}

function MenuLink({ href, onClick, children }: { href: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Link href={href} onClick={onClick} className="block px-4 py-2 text-ink-2 hover:text-ink hover:underline">
      {children}
    </Link>
  );
}

/** A small disclosure menu: opens on click; closes on an outside click, Escape, or choosing an item. */
function Menu({
  trigger,
  label,
  align,
  children,
}: {
  trigger: React.ReactNode;
  label?: string;
  align: "left" | "right";
  children: (close: () => void) => React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onPointer = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-label={label}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded text-ink"
      >
        {trigger}
        <ChevronDown size={16} strokeWidth={2} className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </button>
      {open ? (
        <div
          className={`absolute top-full z-50 mt-4 rounded border border-line-strong bg-black/90 shadow-[0_8px_24px_rgb(0_0_0/0.5)] ${
            align === "right" ? "right-0" : "left-0"
          }`}
        >
          {children(() => setOpen(false))}
        </div>
      ) : null}
    </div>
  );
}
