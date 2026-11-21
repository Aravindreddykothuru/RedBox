"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Backdrop } from "@/components/backdrop";
import { Button, FloatingInput } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { authMode, startDemoSession, startLocalSession, type TokenResponse } from "@/lib/session";
import { supabase } from "@/lib/supabase";

function safeNext(next: string | null) {
  // only same-site paths, never an absolute URL (open-redirect guard)
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/home";
}

/** For the plain inputs on the account page. */
export const inputClass =
  "h-11 w-full rounded border border-[rgb(128_128_128/0.7)] bg-[rgb(22_22_22/0.7)] px-3 text-ink placeholder:text-muted focus:border-ink";

type Mode = "signin" | "signup";

export function LoginView() {
  const router = useRouter();
  const params = useSearchParams();
  const next = safeNext(params.get("next"));
  const done = () => router.push(next);
  return (
    <div className="relative isolate -mt-16 ml-[calc(50%-50vw)] flex min-h-svh w-screen items-center justify-center px-4 pb-16 pt-28 sm:px-8">
      <Backdrop />
      <div className="enter w-full max-w-[450px] rounded bg-black/75 px-6 py-12 sm:px-16">
        {authMode === "demo" ? (
          <DemoForm onDone={done} />
        ) : (
          <AccountForm
            onDone={done}
            initialMode={params.get("mode") === "signup" ? "signup" : "signin"}
            initialEmail={params.get("email") ?? ""}
          />
        )}
      </div>
    </div>
  );
}

function DemoForm({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        startDemoSession(name);
        onDone();
      }}
    >
      <h1 className="text-[32px] font-bold leading-10">Create a profile</h1>
      <p className="mt-2 text-sm text-ink-2">Demo mode: your profile is kept in this browser, no password needed</p>
      <FloatingInput id="name" label="Your name" className="mt-6" value={name} maxLength={80} autoComplete="nickname" onChange={(e) => setName(e.target.value)} required />
      <Button type="submit" size="lg" className="mt-6 w-full">
        Continue
      </Button>
    </form>
  );
}

/** Email + password: our own accounts ("local"), or Supabase Auth. */
function AccountForm({ onDone, initialMode, initialEmail }: { onDone: () => void; initialMode: Mode; initialEmail: string }) {
  const { t } = useI18n();
  const [mode, setMode] = useState<Mode>(initialMode);
  const [email, setEmail] = useState(initialEmail);
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ kind: "error" | "info"; text: string } | null>(null);

  async function submitLocal() {
    const token =
      mode === "signin"
        ? await api<TokenResponse>("/auth/login", { method: "POST", body: { email, password } })
        : await api<TokenResponse>("/auth/register", { method: "POST", body: { email, password, display_name: name || null } });
    startLocalSession(token);
    onDone();
  }

  async function submitSupabase() {
    if (!supabase) throw new Error("Supabase is not configured");
    if (mode === "signin") {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      onDone();
    } else {
      const { data, error } = await supabase.auth.signUp({ email, password, options: { data: { display_name: name } } });
      if (error) throw error;
      if (data.session) onDone();
      else setMessage({ kind: "info", text: "Check your inbox and confirm your email, then sign in." });
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      await (authMode === "supabase" ? submitSupabase() : submitLocal());
    } catch (err) {
      const text =
        err instanceof ApiError && err.status === 422
          ? "Check the email address, and use a password of at least 8 characters."
          : err instanceof Error
            ? err.message
            : "Sign-in failed";
      setMessage({ kind: "error", text: text.charAt(0).toUpperCase() + text.slice(1) });
    } finally {
      setBusy(false);
    }
  }

  const switchTo = (m: Mode) => {
    setMode(m);
    setMessage(null);
  };

  return (
    <form onSubmit={submit}>
      <h1 className="text-[32px] font-bold leading-10">{mode === "signin" ? t("nav.signIn") : t("login.create")}</h1>
      {message ? (
        <p
          role="status"
          className={`mt-6 rounded px-4 py-3 text-sm ${message.kind === "error" ? "bg-danger/20 text-ink" : "bg-white/10 text-ink"}`}
        >
          {message.text}
        </p>
      ) : null}
      <div className="mt-6 space-y-4">
        {mode === "signup" ? (
          <FloatingInput id="name" label={t("login.name")} value={name} maxLength={80} autoComplete="nickname" onChange={(e) => setName(e.target.value)} />
        ) : null}
        <FloatingInput id="email" label={t("login.email")} type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <FloatingInput
          id="password"
          label={t("login.password")}
          type="password"
          minLength={8}
          maxLength={128}
          autoComplete={mode === "signin" ? "current-password" : "new-password"}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </div>
      {mode === "signup" ? <p className="mt-2 text-xs text-ink-2">{t("login.pwHint")}</p> : null}
      <Button type="submit" size="lg" className="mt-6 w-full" disabled={busy}>
        {busy ? t("login.wait") : mode === "signin" ? t("nav.signIn") : t("login.createBtn")}
      </Button>
      <p className="mt-8 text-base text-muted">
        {mode === "signin" ? t("login.newTo") : t("login.have")}
        <button
          type="button"
          onClick={() => switchTo(mode === "signin" ? "signup" : "signin")}
          className="font-semibold text-ink hover:underline"
        >
          {mode === "signin" ? t("login.createLink") : t("nav.signIn")}
        </button>
      </p>
      <p className="mt-4 text-[13px] leading-5 text-muted">
        {mode === "signup" && authMode === "local"
          ? "The name is shown only to you. Your password is stored only as a salted hash."
          : "Your ratings stay with your account. You can download or delete them at any time."}
      </p>
    </form>
  );
}
