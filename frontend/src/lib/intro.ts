// The opening animation (components/intro.tsx) plays once per browser tab. This script runs in <head>
// before the first paint, so the page never flashes before the intro, and sets <html data-intro>:
//   "play"  first visit in this tab
//   "skip"  already played in this tab, the visitor asks for reduced motion, or storage is blocked
//   "done"  set by the intro when it ends
// Without JavaScript the attribute is never set and the intro stays hidden (globals.css).
// Kept out of the "use client" component so the server layout can import the string itself.
export const INTRO_KEY = "redbox-intro-seen";

export const INTRO_SCRIPT = `try{var d=document.documentElement,s=sessionStorage;d.dataset.intro=s.getItem("${INTRO_KEY}")||matchMedia("(prefers-reduced-motion: reduce)").matches?"skip":"play";s.setItem("${INTRO_KEY}","1")}catch(e){document.documentElement.dataset.intro="skip"}`;
