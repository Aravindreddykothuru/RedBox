"use client";

// Who is using the app. Three modes (NEXT_PUBLIC_AUTH_MODE), matching the API's REDBOX_AUTH_MODE:
// - "local" (default): email + password accounts handled by our API; the browser keeps the
//   API's signed token and sends it as a Bearer token.
// - "supabase": Supabase Auth accounts (needs NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY).
// - "demo": no accounts; a name and a random id kept in this browser (API in "dev" mode). For local
//   experiments only.

import { useSyncExternalStore } from "react";
import type { Session as SupabaseSession } from "@supabase/supabase-js";
import { API_URL } from "./config";
import { supabase } from "./supabase";

export type AuthMode = "local" | "supabase" | "demo";

export type Session = {
  mode: AuthMode;
  userId: string;
  name: string;
  email?: string;
  accessToken?: string;
  isAdmin?: boolean;
};

type State = { ready: boolean; session: Session | null };

const configured = process.env.NEXT_PUBLIC_AUTH_MODE as AuthMode | undefined;
export const authMode: AuthMode = configured ?? (supabase ? "supabase" : "local");

const LOCAL_KEY = "redbox.session";
const DEMO_KEY = "redbox.demo-session";
const SERVER_STATE: State = { ready: false, session: null };
let state: State = SERVER_STATE;
let started = false;
const listeners = new Set<() => void>();

function set(next: State) {
  state = next;
  listeners.forEach((l) => l());
}

function readStored(key: string): Session | null {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

function store(key: string, session: Session | null) {
  try {
    if (session) window.localStorage.setItem(key, JSON.stringify(session));
    else window.localStorage.removeItem(key);
  } catch {
    // storage unavailable (private mode): the session lasts until the tab closes
  }
}

function fromSupabase(s: SupabaseSession | null): Session | null {
  if (!s) return null;
  const meta = s.user.user_metadata ?? {};
  return {
    mode: "supabase",
    userId: s.user.id,
    name: (meta.display_name as string) || s.user.email?.split("@")[0] || "you",
    email: s.user.email ?? undefined,
    accessToken: s.access_token,
  };
}

/** Local accounts: confirm the stored token with the API (it may have expired or been revoked). */
async function refreshLocal(session: Session) {
  try {
    const res = await fetch(`${API_URL}/auth/me`, { headers: { Authorization: `Bearer ${session.accessToken}` } });
    if (res.status === 401) return endSession();
    if (res.ok) {
      const me = (await res.json()) as { display_name: string | null; email: string | null; is_admin: boolean };
      const next = { ...session, name: me.display_name || session.name, email: me.email ?? session.email, isAdmin: me.is_admin };
      store(LOCAL_KEY, next);
      set({ ready: true, session: next });
    }
  } catch {
    // API unreachable: keep the session; requests will show their own error
  }
}

function start() {
  if (started || typeof window === "undefined") return;
  started = true;
  if (authMode === "supabase" && supabase) {
    supabase.auth.getSession().then(({ data }) => set({ ready: true, session: fromSupabase(data.session) }));
    supabase.auth.onAuthStateChange((_event, s) => set({ ready: true, session: fromSupabase(s) }));
  } else if (authMode === "local") {
    const saved = readStored(LOCAL_KEY);
    set({ ready: true, session: saved });
    if (saved) void refreshLocal(saved);
  } else {
    set({ ready: true, session: readStored(DEMO_KEY) });
  }
}

const sessionStore = {
  subscribe(listener: () => void) {
    listeners.add(listener);
    start();
    return () => listeners.delete(listener);
  },
  get: () => state,
  server: () => SERVER_STATE,
};

export function useSession(): State {
  return useSyncExternalStore(sessionStore.subscribe, sessionStore.get, sessionStore.server);
}

export function currentSession(): Session | null {
  return state.session;
}

export type TokenResponse = {
  access_token: string;
  user: { user_id: string; email: string | null; display_name: string | null; is_admin: boolean };
};

/** Local accounts: store what /auth/login or /auth/register returned. */
export function startLocalSession(token: TokenResponse): Session {
  const session: Session = {
    mode: "local",
    userId: token.user.user_id,
    name: token.user.display_name || token.user.email?.split("@")[0] || "you",
    email: token.user.email ?? undefined,
    accessToken: token.access_token,
    isAdmin: token.user.is_admin,
  };
  store(LOCAL_KEY, session);
  set({ ready: true, session });
  return session;
}

export function updateSessionName(name: string) {
  if (!state.session) return;
  const next = { ...state.session, name };
  store(state.session.mode === "demo" ? DEMO_KEY : LOCAL_KEY, next);
  set({ ready: true, session: next });
}

export function startDemoSession(name: string): Session {
  const existing = readStored(DEMO_KEY);
  const session: Session = {
    mode: "demo",
    userId: existing?.userId ?? crypto.randomUUID(),
    name: name.trim() || "Guest",
  };
  store(DEMO_KEY, session);
  set({ ready: true, session });
  return session;
}

/** Forget the session in this browser (the server side is handled by the caller where needed). */
export function endSession() {
  store(LOCAL_KEY, null);
  store(DEMO_KEY, null);
  set({ ready: true, session: null });
}

export async function signOut() {
  if (authMode === "supabase" && supabase) await supabase.auth.signOut();
  endSession();
}
