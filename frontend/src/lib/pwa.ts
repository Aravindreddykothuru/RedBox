"use client";

import { useEffect, useSyncExternalStore } from "react";

// The browser's install prompt (Chrome, Edge, Android): kept when it fires so the account menu can offer
// "Install app" at a moment the user chooses. Safari has no prompt; there, Share > Add to Home Screen works.
type InstallPrompt = Event & { prompt: () => Promise<void>; userChoice: Promise<{ outcome: string }> };

let deferred: InstallPrompt | null = null;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());

if (typeof window !== "undefined") {
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferred = e as InstallPrompt;
    notify();
  });
  window.addEventListener("appinstalled", () => {
    deferred = null;
    notify();
  });
}

/** True while the browser lets the site be installed; install() shows the browser's own dialog. */
export function useInstallPrompt(): { canInstall: boolean; install: () => Promise<void> } {
  const canInstall = useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => deferred !== null,
    () => false,
  );
  return {
    canInstall,
    install: async () => {
      if (!deferred) return;
      await deferred.prompt();
      await deferred.userChoice;
      deferred = null;
      notify();
    },
  };
}

/** Registers public/sw.js in production builds (a dev server's files change on every save). */
export function ServiceWorker() {
  useEffect(() => {
    if (process.env.NODE_ENV === "production" && "serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }
  }, []);
  return null;
}
