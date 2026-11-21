import { STILLS } from "@/lib/stills";

const ROWS = 6;
// A row drifts by half its length, so half a row must be wider than the screen: about 17 row-heights
// covers the tilted, oversized wall on a 2560px-wide monitor at the tallest row height.
const HALF_ROW_HEIGHTS = 17;
const GAP_IN_ROW_HEIGHTS = 0.06; // the 10px gap at a ~170px row

/** Rows of film stills behind the landing and sign-in heroes, each image whole at its own shape, the wall
 *  tilted a little and every row drifting slowly, alternate rows the other way (CSS in globals.css, "Stills
 *  wall"; still with reduced motion). Row r shows every sixth still starting at r, so every still appears
 *  and neighbouring rows differ; each row repeats its run until half of it covers the screen.
 *  Decorative, so hidden from screen readers. */
export function StillsWall() {
  return (
    <div className="stills-wall">
      {Array.from({ length: ROWS }, (_, r) => {
        const run = STILLS.filter((_, i) => i % ROWS === r);
        const runWidth = run.reduce((sum, s) => sum + s.w / s.h + GAP_IN_ROW_HEIGHTS, 0);
        const copies = 2 * Math.max(1, Math.ceil(HALF_ROW_HEIGHTS / runWidth));
        return (
          <div key={r} className="stills-row">
            <div className="stills-track">
              {Array.from({ length: copies }, (_, c) =>
                run.map((s, k) => (
                  // eslint-disable-next-line @next/next/no-img-element -- small pre-sized local WebPs
                  <img key={`${c}-${k}`} src={s.src} width={s.w} height={s.h} alt="" decoding="async" draggable={false} />
                )),
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
