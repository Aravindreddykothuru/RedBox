"use client";

// Chart rules: one series per chart; the hybrid is the one highlighted mark (brand red), every other model is
// the series blue. Both colours are validated against the dark surface (lightness band, chroma, CVD).
// Text never takes the series colour; bars are <= 24px with a 4px rounded data-end; hairline grids.

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { ResultRow } from "@/lib/results";

const SERIES = "var(--chart-series)";
const HIGHLIGHT = "var(--chart-highlight)";
const GRID = "var(--chart-grid)";
const AXIS = "var(--chart-axis)";
const tick = { fill: "var(--muted)", fontSize: 12 };

type Point = { model: string; isHybrid: boolean; ndcg: number; coverage: number; fit: number };

function points(rows: ResultRow[]): Point[] {
  return rows.map((r) => ({
    model: r.model,
    isHybrid: r.isHybrid,
    ndcg: r.metrics["NDCG@10"] ?? 0,
    coverage: r.metrics["Coverage@10"] ?? 0,
    fit: Math.max(r.metrics["fit (s)"] ?? 0, 0.0001),
  }));
}

function TooltipBox({ title, lines }: { title: string; lines: string[] }) {
  return (
    <div className="rounded-lg border border-line-strong bg-deep px-3 py-2 text-xs">
      <p className="font-medium text-ink">{title}</p>
      {lines.map((l) => (
        <p key={l} className="text-ink-2">
          {l}
        </p>
      ))}
    </div>
  );
}

type TipProps = { active?: boolean; payload?: { payload: Point }[] };

function PointTooltip({ active, payload }: TipProps) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <TooltipBox
      title={p.model}
      lines={[`NDCG@10 ${p.ndcg.toFixed(4)}`, `Coverage@10 ${(p.coverage * 100).toFixed(1)}%`, `Training ${formatSeconds(p.fit)}`]}
    />
  );
}

export function NdcgBars({ rows }: { rows: ResultRow[] }) {
  const data = points(rows).sort((a, b) => b.ndcg - a.ndcg);
  return (
    <div style={{ height: data.length * 34 + 40 }} role="img" aria-label="Bar chart of NDCG@10 per model; the table below has the values">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }} barCategoryGap={8}>
          <CartesianGrid horizontal={false} stroke={GRID} />
          <XAxis type="number" tick={tick} axisLine={{ stroke: AXIS }} tickLine={false} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis type="category" dataKey="model" width={132} tick={tick} axisLine={{ stroke: AXIS }} tickLine={false} />
          <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<PointTooltip />} />
          <Bar dataKey="ndcg" barSize={18} radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.model} fill={d.isHybrid ? HIGHLIGHT : SERIES} />
            ))}
            <LabelList dataKey="ndcg" position="right" fill="var(--ink-2)" fontSize={12} formatter={(v: unknown) => Number(v).toFixed(3)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Direct labels only on the story's marks (hybrid, best single model, the popularity baseline). */
function labelled(data: Point[]): Set<string> {
  const singles = data.filter((d) => !d.isHybrid).sort((a, b) => b.ndcg - a.ndcg);
  return new Set([data.find((d) => d.isHybrid)?.model, singles[0]?.model, "MostPop"].filter(Boolean) as string[]);
}

export function CoverageScatter({ rows }: { rows: ResultRow[] }) {
  const data = points(rows);
  const show = labelled(data);
  const renderDot = (props: { cx?: number; cy?: number; payload?: Point }) => {
    const { cx = 0, cy = 0, payload } = props;
    const isHybrid = payload?.isHybrid;
    return (
      <g>
        <circle cx={cx} cy={cy} r={isHybrid ? 7 : 5} fill={isHybrid ? HIGHLIGHT : SERIES} stroke="var(--surface)" strokeWidth={2} />
        {payload && show.has(payload.model) ? (
          <text x={cx + 10} y={cy + 4} fill="var(--ink-2)" fontSize={12}>
            {payload.model}
          </text>
        ) : null}
      </g>
    );
  };
  return (
    <div style={{ height: 340 }} role="img" aria-label="Scatter plot of NDCG@10 against catalogue coverage per model">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 96, bottom: 24, left: 8 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis
            type="number"
            dataKey="coverage"
            name="Coverage@10"
            domain={[0, 1]}
            tick={tick}
            axisLine={{ stroke: AXIS }}
            tickLine={false}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            label={{ value: "Catalogue coverage@10", position: "insideBottom", offset: -14, fill: "var(--muted)", fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="ndcg"
            name="NDCG@10"
            domain={[0, (max: number) => Math.ceil(max * 20) / 20]}
            tickCount={5}
            allowDecimals
            tick={tick}
            axisLine={{ stroke: AXIS }}
            tickLine={false}
            width={44}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <ZAxis range={[80, 80]} />
          <Tooltip cursor={{ stroke: AXIS }} content={<PointTooltip />} />
          <Scatter data={data} shape={renderDot} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function formatSeconds(s: number) {
  if (s < 0.01) return `${(s * 1000).toFixed(1)} ms`;
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 120) return `${s.toFixed(1)} s`;
  return `${(s / 60).toFixed(0)} min`;
}

export function TrainingTimeBars({ rows }: { rows: ResultRow[] }) {
  const data = points(rows).sort((a, b) => b.fit - a.fit);
  return (
    <div style={{ height: data.length * 30 + 40 }} role="img" aria-label="Bar chart of training time per model on a log scale">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 72, bottom: 4, left: 8 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} stroke={GRID} />
          <XAxis
            type="number"
            dataKey="fit"
            scale="log"
            domain={[0.0001, 10000]}
            ticks={[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
            allowDataOverflow
            tick={tick}
            axisLine={{ stroke: AXIS }}
            tickLine={false}
            tickFormatter={(v: number) =>
              v < 1 ? `${Number((v * 1000).toPrecision(2))} ms` : v >= 1000 ? `${v / 1000}k s` : `${v} s`
            }
          />
          <YAxis type="category" dataKey="model" width={132} tick={tick} axisLine={{ stroke: AXIS }} tickLine={false} />
          <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<PointTooltip />} />
          <Bar dataKey="fit" barSize={16} radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.model} fill={d.isHybrid ? HIGHLIGHT : SERIES} />
            ))}
            <LabelList dataKey="fit" position="right" fill="var(--ink-2)" fontSize={12} formatter={(v: unknown) => formatSeconds(Number(v))} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
