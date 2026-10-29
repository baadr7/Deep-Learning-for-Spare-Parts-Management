import { STATES } from "../lib/states";

export function StateLegend() {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-2 text-[13px] text-ink-soft">
      {STATES.map((s) => (
        <li key={s.label} className="flex items-center gap-2">
          <span
            aria-hidden
            className="inline-block size-3 rounded-[3px] border"
            style={{ background: s.fill, borderColor: s.stroke }}
          />
          {s.label}
        </li>
      ))}
    </ul>
  );
}
