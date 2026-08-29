import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

interface TurnSnapshot {
  turnId: string;
  stageMsSinceStart: [string, number][];
  speechToSttFinalMs: number | null;
  sttFinalToRouterDecisionMs: number | null;
  sttFinalToLlmFirstTokenMs: number | null;
  llmFirstTokenToTtsFirstAudioMs: number | null;
  speechEndToFirstAudioMs: number | null;
  totalTurnLatencyMs: number | null;
  tier: string;
  mode: string;
  pressure: string;
  recordedAtMs: number;
  interrupted: boolean;
}

const RECENT_COUNT = 20;

function fmtMs(ms: number | null): string {
  return ms === null ? "N/A" : `${ms} ms`;
}

function fmtTime(ms: number): string {
  return new Date(ms).toLocaleTimeString();
}

/// Nearest-rank percentile over a sorted ascending array — no interpolation,
/// no library. `p` is 0-100. Returns `null` only when the input is empty
/// (never fabricates a value from zero samples).
function percentile(sortedAsc: number[], p: number): number | null {
  if (sortedAsc.length === 0) return null;
  const rank = Math.ceil((p / 100) * sortedAsc.length);
  const index = Math.min(Math.max(rank - 1, 0), sortedAsc.length - 1);
  return sortedAsc[index];
}

interface Stat {
  avg: number | null;
  p50: number | null;
  p95: number | null;
  n: number;
}

/// Computes avg/P50/P95 from ONLY the non-null values of `pick` across
/// `turns`, after `turns` has already been filtered to whichever population
/// (all / native / llm) the caller wants. Never includes a missing
/// measurement as 0 and never mixes populations silently.
function computeStat(turns: TurnSnapshot[], pick: (t: TurnSnapshot) => number | null): Stat {
  const values = turns.map(pick).filter((v): v is number => v !== null);
  if (values.length === 0) return { avg: null, p50: null, p95: null, n: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const avg = Math.round(values.reduce((sum, v) => sum + v, 0) / values.length);
  return { avg, p50: percentile(sorted, 50), p95: percentile(sorted, 95), n: values.length };
}

function StatRow({ label, stat }: { label: string; stat: Stat }) {
  return (
    <div className="latency-stat-row">
      <span className="latency-stat-label">{label}</span>
      {stat.n === 0 ? (
        <span className="latency-stat-values">N/A (n=0)</span>
      ) : (
        <span className="latency-stat-values">
          avg {stat.avg}ms &middot; P50 {stat.p50}ms &middot; P95 {stat.p95}ms <span className="latency-n">(n={stat.n})</span>
        </span>
      )}
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  mic_detected: "Mic detected",
  speech_started: "Speech started",
  speech_ended: "Speech ended",
  stt_started: "STT started",
  stt_first_result: "STT first result",
  stt_final: "STT final",
  router_started: "Router started",
  router_decision: "Router decision",
  llm_started: "LLM started",
  llm_first_token: "LLM first token",
  llm_complete: "LLM complete",
  tts_started: "TTS started",
  tts_first_audio: "TTS first audio",
  playback_started: "Playback started",
  turn_complete: "Turn complete",
};

/// Reads the SAME `TurnSnapshot[]` the backend's latency dashboard commands
/// return (see hardware::telemetry::TurnHistory / hardware::commands) — no
/// second telemetry system, no estimation. Every duration shown here is
/// either a real measured value from that array or the literal string
/// "N/A"; there is no code path that substitutes a different timestamp,
/// assumes a stage took 0ms, or averages across a mismatched population
/// without an attached n= count. All stats are computed from the exact same
/// array the "Recent 20 turns" table renders, so any number can be checked
/// by eye against the raw rows.
export function LatencyDashboardPanel() {
  const [history, setHistory] = useState<TurnSnapshot[]>([]);
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const turns = await invoke<TurnSnapshot[]>("get_turn_telemetry_history");
      setHistory(turns);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleClear = useCallback(async () => {
    setError(null);
    try {
      await invoke("clear_turn_telemetry");
      setSelectedTurnId(null);
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  }, [refresh]);

  const handleExport = useCallback(() => {
    const json = JSON.stringify(history, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `veronica-turn-telemetry-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [history]);

  const recent = useMemo(() => history.slice(0, RECENT_COUNT), [history]);
  const latest = history[0] ?? null;

  const validTurns = useMemo(() => history.filter((t) => !t.interrupted), [history]);
  const interruptedCount = history.length - validTurns.length;

  // "Native/fast-router" turns decided without ever calling an LLM; "LLM/
  // agent" turns did. Classified only from fields already present in the
  // snapshot (never a new heuristic on top of timing) — see
  // hardware::telemetry's doc: a fast-router turn never marks
  // LlmFirstToken/TtsFirstAudio at all.
  const nativeTurns = useMemo(
    () => validTurns.filter((t) => t.sttFinalToLlmFirstTokenMs === null && t.llmFirstTokenToTtsFirstAudioMs === null),
    [validTurns]
  );
  const llmTurns = useMemo(
    () => validTurns.filter((t) => t.sttFinalToLlmFirstTokenMs !== null || t.llmFirstTokenToTtsFirstAudioMs !== null),
    [validTurns]
  );

  const totalLatencyStat = computeStat(validTurns, (t) => t.totalTurnLatencyMs);
  const speechToFirstAudioStat = computeStat(validTurns, (t) => t.speechEndToFirstAudioMs);
  const nativeTotalStat = computeStat(nativeTurns, (t) => t.totalTurnLatencyMs);
  const llmTotalStat = computeStat(llmTurns, (t) => t.totalTurnLatencyMs);

  const STAGE_STATS: { key: keyof TurnSnapshot; label: string }[] = [
    { key: "speechToSttFinalMs", label: "Speech end → STT final" },
    { key: "sttFinalToRouterDecisionMs", label: "STT final → Router decision" },
    { key: "sttFinalToLlmFirstTokenMs", label: "STT final → LLM first token" },
    { key: "llmFirstTokenToTtsFirstAudioMs", label: "LLM first token → TTS first audio" },
  ];

  const bottleneck = useMemo(() => {
    const perStage = STAGE_STATS.map(({ key, label }) => ({
      label,
      stat: computeStat(validTurns, (t) => t[key] as number | null),
    })).filter((s) => s.stat.n > 0);
    if (perStage.length === 0) return null;
    const highest = perStage.reduce((max, s) => ((s.stat.avg ?? -1) > (max.stat.avg ?? -1) ? s : max));
    if (highest.stat.n < 3) return null;
    return highest;
  }, [validTurns]);

  const selectedTurn = history.find((t) => t.turnId === selectedTurnId) ?? null;

  if (loading) {
    return <p className="setup-hint">Loading telemetry…</p>;
  }

  return (
    <div className="personalization-panel latency-dashboard">
      {error && <p className="error">{error}</p>}

      <div className="latency-actions">
        <button className="link-button" style={{ color: "var(--text-muted)" }} onClick={refresh}>
          Refresh
        </button>
        <button className="link-button" style={{ color: "var(--text-muted)" }} onClick={handleExport} disabled={history.length === 0}>
          Export raw telemetry
        </button>
        <button className="link-button" onClick={handleClear} disabled={history.length === 0}>
          Clear telemetry
        </button>
      </div>

      <p className="setup-hint">
        Every number below comes directly from real timestamps recorded by the voice pipeline
        (hardware::telemetry::TurnTelemetry, in-memory, {history.length} turn{history.length === 1 ? "" : "s"} retained,
        {" "}
        {interruptedCount} interrupted/incomplete). "N/A" means that stage was never marked for that turn — it is
        never estimated or assumed to be 0ms.
      </p>

      <section className="latency-section">
        <h4 className="latency-section-title">Latest turn</h4>
        {latest ? (
          <div className="latency-latest">
            <div className="latency-turn-id">turn_id: {latest.turnId}</div>
            <div>Total latency (speech end → turn complete): {fmtMs(latest.totalTurnLatencyMs)}</div>
            <div>Speech end → first audio (measured): {fmtMs(latest.speechEndToFirstAudioMs)}</div>
            {latest.interrupted && <div className="latency-badge-interrupted">Interrupted / incomplete</div>}
          </div>
        ) : (
          <p className="setup-hint">No turns recorded yet.</p>
        )}
      </section>

      <section className="latency-section">
        <h4 className="latency-section-title">Recent {RECENT_COUNT} turns</h4>
        {recent.length === 0 ? (
          <p className="setup-hint">No turns recorded yet.</p>
        ) : (
          <div className="latency-table-wrap">
            <table className="latency-table">
              <thead>
                <tr>
                  <th>Turn</th>
                  <th>Time</th>
                  <th>Total latency</th>
                  <th>Speech→Audio</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((t) => (
                  <tr
                    key={t.turnId}
                    className={["latency-row", t.turnId === selectedTurnId ? "selected" : ""].filter(Boolean).join(" ")}
                    onClick={() => setSelectedTurnId(t.turnId)}
                  >
                    <td className="latency-turn-id-cell">{t.turnId}</td>
                    <td>{fmtTime(t.recordedAtMs)}</td>
                    <td>{fmtMs(t.totalTurnLatencyMs)}</td>
                    <td>{fmtMs(t.speechEndToFirstAudioMs)}</td>
                    <td>{t.interrupted ? <span className="latency-badge-interrupted">interrupted</span> : "ok"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="latency-section">
        <h4 className="latency-section-title">Selected-turn detailed timeline</h4>
        {!selectedTurn ? (
          <p className="setup-hint">Click a row above to see exactly where that turn spent its time.</p>
        ) : (
          <div className="latency-detail">
            <div className="latency-turn-id">
              turn_id: {selectedTurn.turnId} &middot; tier {selectedTurn.tier} &middot; mode {selectedTurn.mode} &middot; pressure{" "}
              {selectedTurn.pressure}
            </div>
            <div className="setup-hint">
              Source: hardware::telemetry::TurnTelemetry (backend, in-memory) — recorded {fmtTime(selectedTurn.recordedAtMs)}
            </div>

            <table className="latency-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>ms since turn start</th>
                </tr>
              </thead>
              <tbody>
                {selectedTurn.stageMsSinceStart.length === 0 ? (
                  <tr>
                    <td colSpan={2} className="setup-hint">
                      No stage timestamps recorded for this turn.
                    </td>
                  </tr>
                ) : (
                  selectedTurn.stageMsSinceStart.map(([stage, ms]) => (
                    <tr key={stage}>
                      <td>{STAGE_LABELS[stage] ?? stage}</td>
                      <td>{ms} ms</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>

            <div className="latency-stat-grid">
              <div>Speech end → STT final: {fmtMs(selectedTurn.speechToSttFinalMs)}</div>
              <div>STT final → Router decision: {fmtMs(selectedTurn.sttFinalToRouterDecisionMs)}</div>
              <div>STT final → LLM first token: {fmtMs(selectedTurn.sttFinalToLlmFirstTokenMs)}</div>
              <div>LLM first token → TTS first audio: {fmtMs(selectedTurn.llmFirstTokenToTtsFirstAudioMs)}</div>
              <div className="latency-stat-total">
                Total (speech end → first audio, measured): {fmtMs(selectedTurn.speechEndToFirstAudioMs)}
              </div>
              <div className="latency-stat-total">Total turn latency: {fmtMs(selectedTurn.totalTurnLatencyMs)}</div>
            </div>
          </div>
        )}
      </section>

      <section className="latency-section">
        <h4 className="latency-section-title">Aggregate statistics</h4>
        <p className="setup-hint">
          Computed only from turns that were not interrupted, and only from turns where that specific stage was
          actually measured. Each line shows its own sample count.
        </p>
        <StatRow label="Total turn latency" stat={totalLatencyStat} />
        <StatRow label="Speech end → first audio" stat={speechToFirstAudioStat} />

        <h5 className="latency-subsection-title">Native/fast-router turns only</h5>
        <StatRow label="Total latency (native)" stat={nativeTotalStat} />

        <h5 className="latency-subsection-title">LLM/agent turns only</h5>
        <StatRow label="Total latency (LLM/agent)" stat={llmTotalStat} />

        <p className="setup-hint">
          Valid measurements: {validTurns.length} &middot; Interrupted/incomplete: {interruptedCount} &middot; Native/fast-router:{" "}
          {nativeTurns.length} &middot; LLM/agent: {llmTurns.length}
        </p>
      </section>

      <section className="latency-section">
        <h4 className="latency-section-title">Bottleneck</h4>
        {bottleneck ? (
          <p>
            Highest measured stage: <strong>{bottleneck.label}</strong> — {bottleneck.stat.avg} ms (n={bottleneck.stat.n})
          </p>
        ) : (
          <p className="setup-hint">Insufficient data for bottleneck analysis.</p>
        )}
      </section>
    </div>
  );
}
