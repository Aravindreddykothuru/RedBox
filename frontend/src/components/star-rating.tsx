"use client";

import { Star } from "lucide-react";
import { useState } from "react";

/** Half-star rating input (0.5 to 5). Click the left half of a star for a half star. Arrow keys adjust,
 *  Delete clears. */
export function StarRating({
  value,
  onChange,
  onClear,
  size = 20,
  label = "Your rating",
  disabled,
}: {
  value: number | null;
  onChange: (v: number) => void;
  onClear?: () => void;
  size?: number;
  label?: string;
  disabled?: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const shown = hover ?? value ?? 0;

  function pick(e: React.MouseEvent<HTMLButtonElement>, star: number) {
    const rect = e.currentTarget.getBoundingClientRect();
    return e.clientX - rect.left < rect.width / 2 ? star - 0.5 : star;
  }

  function onKey(e: React.KeyboardEvent) {
    const current = value ?? 0;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      onChange(Math.min(5, current + 0.5));
    } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      if (current > 0.5) onChange(current - 0.5);
    } else if ((e.key === "Delete" || e.key === "Backspace") && onClear) {
      e.preventDefault();
      onClear();
    }
  }

  return (
    <div
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={label}
      aria-valuemin={0.5}
      aria-valuemax={5}
      aria-valuenow={value ?? undefined}
      aria-valuetext={value ? `${value} of 5 stars` : "not rated"}
      aria-disabled={disabled}
      onKeyDown={disabled ? undefined : onKey}
      onMouseLeave={() => setHover(null)}
      className="inline-flex items-center rounded-md"
    >
      {[1, 2, 3, 4, 5].map((star) => {
        const fill = shown >= star ? 1 : shown >= star - 0.5 ? 0.5 : 0;
        return (
          <button
            key={star}
            type="button"
            tabIndex={-1}
            disabled={disabled}
            aria-hidden
            onMouseMove={(e) => setHover(pick(e, star))}
            onClick={(e) => onChange(pick(e, star))}
            className="relative p-0.5 text-muted/70 disabled:cursor-default"
            style={{ width: size + 4, height: size + 4 }}
          >
            <Star size={size} strokeWidth={1.5} className="absolute left-0.5 top-0.5" />
            {fill > 0 ? (
              <span className="absolute left-0.5 top-0.5 overflow-hidden" style={{ width: fill === 1 ? size : size / 2 }}>
                <Star size={size} strokeWidth={1.5} className="fill-accent text-accent" />
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
