const intFmt = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
const oneFmt = new Intl.NumberFormat("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

export function fmtInt(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "n.d." : intFmt.format(v);
}

export function fmtNum(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "n.d.";
  return new Intl.NumberFormat("fr-FR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(v);
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "n.d.";
  return `${fmtNum(v * 100, digits)} %`;
}

export function fmtUR(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "n.d." : `${intFmt.format(v)} UR`;
}

export function fmtHours(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "n.d.";
  if (v >= 48) return `${oneFmt.format(v / 24)} j`;
  return `${oneFmt.format(v)} h`;
}

/** Moyenne glissante centrée sur `window` jours, en ignorant les valeurs manquantes. */
export function rolling(values: (number | null)[], window = 7): (number | null)[] {
  const half = Math.floor(window / 2);
  return values.map((_, i) => {
    let sum = 0;
    let n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) {
      const v = values[j];
      if (v != null && Number.isFinite(v)) {
        sum += v;
        n++;
      }
    }
    return n ? sum / n : null;
  });
}

export function dayLabel(day: number): string {
  return `J${day + 1}`;
}

export function eventSentence(
  e: { type: string; machine: number | null; component: number | null; part?: number; qty?: number; urgent?: boolean },
  components: string[],
  parts: { code: string; name: string }[],
): string {
  const machine = e.machine != null ? `Machine ${e.machine + 1}` : "";
  const comp = e.component != null ? components[e.component]?.toLowerCase() : "";
  const part = e.part != null ? parts[e.part] : undefined;
  switch (e.type) {
    case "panne":
      return `${machine} : panne, ${comp}`;
    case "rupture":
      return `${machine} attend une pièce ${part?.code ?? ""} (stock vide)`;
    case "maintenance_preventive":
      return `${machine} : maintenance préventive, ${comp}`;
    case "remise_en_service":
      return `${machine} remise en service`;
    case "commande":
      return `Commande ${e.urgent ? "urgente " : ""}de ${e.qty} ${part?.code ?? ""}`;
    case "reception":
      return `Réception de ${e.qty} ${part?.code ?? ""}`;
    case "revision":
      return `${machine} : début de révision générale`;
    case "fin_revision":
      return `${machine} : révision générale terminée`;
    case "fin_preventive":
      return `${machine} : maintenance préventive terminée`;
    case "echec_test":
      return `${machine} : test non conforme, reprise`;
    default:
      return e.type;
  }
}
