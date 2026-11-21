"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { RequireSession } from "@/components/require-session";
import { Button, ButtonLink, PageHeader } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { endSession, signOut, startLocalSession, updateSessionName, type Session, type TokenResponse } from "@/lib/session";
import { inputClass } from "../login/login-view";
import { BRAND, BRAND_SLUG } from "@/lib/brand";

export function AccountView() {
  return <RequireSession why="to manage an account">{(s) => <Account session={s} />}</RequireSession>;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-line bg-surface p-6">
      <h2 className="font-medium">{title}</h2>
      <div className="mt-3">{children}</div>
    </section>
  );
}

function Status({ msg }: { msg: { ok: boolean; text: string } | null }) {
  if (!msg) return null;
  return (
    <p role="status" className={`mt-3 text-sm ${msg.ok ? "text-ink-2" : "text-danger"}`}>
      {msg.text}
    </p>
  );
}

function errorText(e: unknown) {
  if (e instanceof ApiError && e.status === 422) return "Use at least 8 characters for the new password.";
  return e instanceof Error ? e.message.charAt(0).toUpperCase() + e.message.slice(1) : "Something went wrong.";
}

function Account({ session }: { session: Session }) {
  const router = useRouter();
  const qc = useQueryClient();
  const local = session.mode === "local";

  const [name, setName] = useState(session.name);
  const [nameMsg, setNameMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [current, setCurrent] = useState("");
  const [fresh, setFresh] = useState("");
  const [pwMsg, setPwMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [confirm, setConfirm] = useState("");
  const [deleteMsg, setDeleteMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  async function run(key: string, fn: () => Promise<void>) {
    setBusy(key);
    try {
      await fn();
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="max-w-2xl">
      <PageHeader title="Your account">{session.email ? `Signed in as ${session.email}` : "Your profile in this browser."}</PageHeader>
      <div className="space-y-6">
        <Section title="Name">
          <form
            className="flex flex-wrap gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              void run("name", async () => {
                try {
                  await api("/me/profile", { method: "PATCH", body: { display_name: name } });
                  updateSessionName(name.trim() || session.name);
                  setNameMsg({ ok: true, text: "Saved." });
                } catch (err) {
                  setNameMsg({ ok: false, text: errorText(err) });
                }
              });
            }}
          >
            <input aria-label="Name" className={`${inputClass} max-w-xs`} value={name} maxLength={80} onChange={(e) => setName(e.target.value)} />
            <Button type="submit" variant="secondary" disabled={busy === "name"}>
              Save
            </Button>
          </form>
          <Status msg={nameMsg} />
        </Section>

        {local ? (
          <Section title="Password">
            <form
              className="grid max-w-sm gap-3"
              onSubmit={(e) => {
                e.preventDefault();
                void run("password", async () => {
                  try {
                    const token = await api<TokenResponse>("/auth/password", {
                      method: "POST",
                      body: { current_password: current, new_password: fresh },
                    });
                    startLocalSession(token); // this device stays signed in; every other one is signed out
                    setCurrent("");
                    setFresh("");
                    setPwMsg({ ok: true, text: "Password changed. Other devices have been signed out." });
                  } catch (err) {
                    setPwMsg({ ok: false, text: errorText(err) });
                  }
                });
              }}
            >
              <input aria-label="Current password" type="password" autoComplete="current-password" placeholder="Current password" className={inputClass} value={current} onChange={(e) => setCurrent(e.target.value)} required />
              <input aria-label="New password" type="password" autoComplete="new-password" minLength={8} placeholder="New password (8+ characters)" className={inputClass} value={fresh} onChange={(e) => setFresh(e.target.value)} required />
              <Button type="submit" variant="secondary" disabled={busy === "password"}>
                Change password
              </Button>
            </form>
            <Status msg={pwMsg} />
          </Section>
        ) : null}

        <Section title="Watch history">
          <p className="text-sm text-ink-2">
            Bring in what you have watched on Netflix, IMDb or Letterboxd, so your picks start from your real taste.
            Imported films are listed in your library, where you can remove them.
          </p>
          <ButtonLink href="/import" variant="secondary" className="mt-3">
            Import history
          </ButtonLink>
        </Section>

        <Section title="Your data">
          <p className="text-sm text-ink-2">Download everything {BRAND} stores about you: ratings, watchlist, imported history, profile and activity</p>
          <Button
            variant="secondary"
            className="mt-3"
            disabled={busy === "export"}
            onClick={() =>
              void run("export", async () => {
                const data = await api<unknown>("/me/export");
                const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
                const a = document.createElement("a");
                a.href = url;
                a.download = `${BRAND_SLUG}-my-data.json`;
                a.click();
                URL.revokeObjectURL(url);
              })
            }
          >
            Download my data
          </Button>
        </Section>

        {local ? (
          <Section title="Sign out everywhere">
            <p className="text-sm text-ink-2">Ends every session, on every device, including this one</p>
            <Button
              variant="secondary"
              className="mt-3"
              disabled={busy === "logout"}
              onClick={() =>
                void run("logout", async () => {
                  await api("/auth/logout-all", { method: "POST" });
                  endSession();
                  qc.clear();
                  router.push("/login");
                })
              }
            >
              Sign out everywhere
            </Button>
          </Section>
        ) : null}

        <Section title="Delete account">
          <p className="text-sm text-ink-2">Permanently deletes your account, ratings, watchlist and activity. This cannot be undone.</p>
          <form
            className="mt-3 flex flex-wrap gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              void run("delete", async () => {
                try {
                  await api("/me/delete", { method: "POST", body: { password: confirm } });
                  await signOut();
                  qc.clear();
                  router.push("/");
                } catch (err) {
                  setDeleteMsg({ ok: false, text: errorText(err) });
                }
              });
            }}
          >
            {local ? (
              <input aria-label="Password to confirm" type="password" autoComplete="current-password" placeholder="Your password" className={`${inputClass} max-w-xs`} value={confirm} onChange={(e) => setConfirm(e.target.value)} required />
            ) : null}
            <button type="submit" disabled={busy === "delete"} className="h-10 rounded-lg border border-danger/60 px-4 text-sm font-medium text-danger hover:bg-danger/10 disabled:opacity-50">
              Delete my account
            </button>
          </form>
          <Status msg={deleteMsg} />
        </Section>
      </div>
    </div>
  );
}
