export interface MeetingSummary {
  summary: string;
  key_points: string[];
  decisions: string[];
  action_items: string[];
  next_steps: string[];
  message: string;
}

function ListSection({ label, items }: { label: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="setup-focus-group">
      <span className="setup-focus-label">{label}</span>
      <ul className="question-list large">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export function MeetingSummaryView({ summary }: { summary: MeetingSummary }) {
  return (
    <div className="setup-focus" style={{ maxHeight: "none" }}>
      <div className="setup-focus-head">
        <span className="setup-focus-title">Meeting Summary</span>
      </div>
      <div className="setup-focus-body">
        {summary.summary && <p className="setup-focus-line">{summary.summary}</p>}
        <ListSection label="Key Points" items={summary.key_points} />
        <ListSection label="Decisions" items={summary.decisions} />
        <ListSection label="Action Items" items={summary.action_items} />
        <ListSection label="Next Steps" items={summary.next_steps} />
      </div>
    </div>
  );
}
