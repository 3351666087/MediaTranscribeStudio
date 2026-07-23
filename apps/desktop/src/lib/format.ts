export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatMilliseconds(milliseconds: number): string {
  const totalSeconds = milliseconds / 1000;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds]
    .map((part, index) =>
      index < 2 ? String(Math.floor(part)).padStart(2, "0") : part.toFixed(3).padStart(6, "0"),
    )
    .join(":");
}

export function cx(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}
