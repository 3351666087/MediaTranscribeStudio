export function ProgressRing({
  value,
  size = 92,
  label,
}: {
  value: number;
  size?: number;
  label: string;
}) {
  const normalized = Math.min(100, Math.max(0, value));
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (normalized / 100) * circumference;

  return (
    <div className="progress-ring" style={{ width: size, height: size }}>
      <svg viewBox="0 0 100 100" role="img" aria-label={`${label} ${normalized}%`}>
        <circle className="progress-ring__track" cx="50" cy="50" r={radius} />
        <circle
          className="progress-ring__value"
          cx="50"
          cy="50"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <span className="progress-ring__label" aria-hidden="true">
        <strong>{normalized}</strong>
        <small>%</small>
      </span>
    </div>
  );
}
