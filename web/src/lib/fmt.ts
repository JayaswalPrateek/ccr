/**
 * Abbreviate a financial number to K / M / B.
 * e.g. 1_250_000 → "1.25M", 85_000 → "85K", 940 → "940"
 *
 * @param v        The raw number
 * @param decimals Max decimal places when no abbreviation applies (default 2)
 */
export function fmtNum(v: number, decimals = 2): string {
  if (!isFinite(v)) return '—';
  const abs = Math.abs(v);
  if (abs >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (abs >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return (v / 1e3).toFixed(1) + 'K';
  return v.toLocaleString(undefined, { maximumFractionDigits: decimals });
}
