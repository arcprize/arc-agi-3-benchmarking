import React, { useEffect, useMemo, useState } from "react";
import type {
  AgentState,
  BreakpointPoint,
  BreakpointStep,
  GlobalState,
  ServerMessage,
} from "./types";

type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

const BREAKPOINT_STEPS: BreakpointStep[] = [
  "analyze",
  "decide",
  "convert",
  "execute_action",
];

const BREAKPOINT_POINTS: BreakpointPoint[] = [
  "analyze_post",
  "decide_post",
  "convert_post",
  "execute_action_pre",
  "execute_action_post",
];

const STEP_LABEL: Record<BreakpointStep, string> = {
  analyze: "Analyze",
  decide: "Decide",
  convert: "Convert",
  execute_action: "Execute",
};

const POINT_LABEL: Record<BreakpointPoint, string> = {
  analyze_post: "Analyze",
  decide_post: "Decide",
  convert_post: "Convert",
  execute_action_pre: "Execute (Pre)",
  execute_action_post: "Execute (Post)",
};

const formatPointLabel = (point: string): string => {
  // Most values are breakpoint points, but sometimes we show other step labels like "play_game".
  if (point in POINT_LABEL) return POINT_LABEL[point as BreakpointPoint];
  // Always return uppercase for display
  return point.toUpperCase();
};

const BREAKPOINT_ROWS: Array<{ point: BreakpointPoint; label: string }> = [
  { point: "analyze_post", label: "Analyze" },
  { point: "decide_post", label: "Decide" },
  { point: "convert_post", label: "Convert" },
  { point: "execute_action_pre", label: "Execute (Pre)" },
  { point: "execute_action_post", label: "Execute (Post)" },
];

export const App: React.FC = () => {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [agents, setAgents] = useState<Record<string, AgentState>>({});
  const [globalState, setGlobalState] = useState<GlobalState>({
    breakpoints: {
      analyze_post: true,
      decide_post: true,
      convert_post: true,
      execute_action_pre: true,
      execute_action_post: true,
    },
  });
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [breakpointForm, setBreakpointForm] = useState<any | null>(null);

  // Determine active breakpoint for the currently selected agent
  const selectedAgent: AgentState | null = useMemo(() => {
    if (!selectedAgentId) return null;
    return agents[selectedAgentId] || null;
  }, [selectedAgentId, agents]);

  // Connect to WebSocket server
  useEffect(() => {
    const url = `ws://${window.location.hostname}:8765/ws`;
    const socket = new WebSocket(url);
    setConnectionStatus("connecting");

    socket.onopen = () => {
      setConnectionStatus("connected");
      // Initial handshake: identify as UI
      socket.send(JSON.stringify({ client: "ui" }));
    };

    socket.onclose = () => {
      setConnectionStatus("disconnected");
    };

    socket.onerror = () => {
      setConnectionStatus("error");
    };

    socket.onmessage = (event) => {
      try {
        const msg: ServerMessage = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch {
        // Ignore malformed messages
      }
    };

    setWs(socket);

    return () => {
      socket.close();
      setWs(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleServerMessage = (msg: ServerMessage) => {
    switch (msg.type) {
      case "state_snapshot": {
        const nextAgents: Record<string, AgentState> = {};
        for (const a of msg.agents) {
          nextAgents[a.agent_id] = a;
        }
        setAgents(nextAgents);
        setGlobalState(msg.global);
        break;
      }
      case "agent_updated": {
        setAgents((prev) => {
          // Always update if agent exists in our state, or if it's not disconnected
          // This ensures we show disconnected agents that were previously connected
          if (msg.agent.agent_id in prev || msg.agent.status !== "DISCONNECTED") {
            return {
          ...prev,
          [msg.agent.agent_id]: msg.agent,
            };
          }
          return prev;
        });
        // If selected agent transitioned out of breakpoint, clear form
        if (
          selectedAgentId === msg.agent.agent_id &&
          (!msg.agent.current_breakpoint || msg.agent.status !== "PAUSED")
        ) {
          setBreakpointForm(null);
        }
        break;
      }
      case "breakpoint_pending": {
        setAgents((prev) => ({
          ...prev,
          [msg.agent.agent_id]: msg.agent,
        }));
        // Pre-fill form when breakpoint becomes pending
        if (selectedAgentId === msg.agent.agent_id) {
          setBreakpointForm(msg.payload);
        }
        break;
      }
      case "breakpoint_resolved": {
        // Only clear if it's the currently-selected agent's current request
        if (selectedAgentId === msg.agent_id) {
        setBreakpointForm(null);
        }
        break;
      }
      case "global_state_updated": {
        setGlobalState(msg.global);
        break;
      }
      case "agent_removed": {
        setAgents((prev) => {
          const next = { ...prev };
          delete next[msg.agent_id];
          return next;
        });
        // Clear selection if removed agent was selected
        if (selectedAgentId === msg.agent_id) {
          setSelectedAgentId(null);
          setBreakpointForm(null);
        }
        break;
      }
    }
  };

  const send = (payload: any) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(payload));
  };

  const handleToggleGlobalPoint = (point: BreakpointPoint) => {
    const next = {
      ...globalState.breakpoints,
      [point]: !globalState.breakpoints[point],
    };
    send({
      type: "set_global_state",
      breakpoints: next,
    });
  };

  const handleSetAllGlobalBreakpoints = (value: boolean) => {
    const next: Record<BreakpointPoint, boolean> = { ...globalState.breakpoints };
    for (const p of BREAKPOINT_POINTS) next[p] = value;
    send({
      type: "set_global_state",
      breakpoints: next,
    });
  };

  const handleToggleAgentPoint = (agent: AgentState, point: BreakpointPoint) => {
    const next = {
      ...agent.breakpoints,
      [point]: !agent.breakpoints[point],
    };
    send({
      type: "set_agent_breakpoints",
      agent_id: agent.agent_id,
      breakpoints: next,
    });
  };

  const handleSetAllAgentBreakpoints = (agent: AgentState, value: boolean) => {
    const next: Record<BreakpointPoint, boolean> = { ...agent.breakpoints };
    for (const p of BREAKPOINT_POINTS) next[p] = value;
    send({
      type: "set_agent_breakpoints",
      agent_id: agent.agent_id,
      breakpoints: next,
    });
  };

  const handleContinueAgent = (agentId: string) => {
    const agent = agents[agentId];
    // Only continue if agent is actually paused at a breakpoint
    if (
      agent &&
      agent.status === "PAUSED" &&
      agent.current_breakpoint &&
      agent.current_breakpoint.request_id
    ) {
      // Optimistically update UI so agent doesn't look stuck
      setAgents((prev) => ({
        ...prev,
        [agentId]: {
          ...agent,
          status: "RUNNING",
          current_breakpoint: null,
        },
      }));
      setBreakpointForm(null);
      send({
        type: "continue_request",
        agent_id: agentId,
        request_id: agent.current_breakpoint.request_id,
      });
    }
  };

  const handleRemoveAgent = (agentId: string) => {
    send({ type: "remove_agent", agent_id: agentId });
  };

  const handleContinueAll = () => {
    send({ type: "continue_all" });
  };

  const handleContinueWithOverrides = () => {
    if (!selectedAgent || !selectedAgent.current_breakpoint) return;
    if (!selectedAgent.current_breakpoint.request_id) return;
    // Optimistically update UI
    setAgents((prev) => ({
      ...prev,
      [selectedAgent.agent_id]: {
        ...selectedAgent,
        status: "RUNNING",
        current_breakpoint: null,
      },
    }));
    setBreakpointForm(null);
    send({
      type: "continue_request",
      agent_id: selectedAgent.agent_id,
      request_id: selectedAgent.current_breakpoint.request_id,
      payload: breakpointForm ?? selectedAgent.current_breakpoint.payload,
    });
  };

  const setFormField = (key: string, value: any) => {
    const base = breakpointForm ?? selectedAgent?.current_breakpoint?.payload ?? {};
    setBreakpointForm({ ...base, [key]: value });
  };

  const copyJson = async (obj: any) => {
    const text = JSON.stringify(obj, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        return true;
      } catch {
        return false;
        }
      }
  };

  const renderImagePreview = () => {
    if (!selectedAgent?.current_breakpoint) return null;
    const payload = breakpointForm ?? selectedAgent.current_breakpoint.payload;
    const imageData: string | null = payload.latest_frame_image?.data ?? null;

    if (!imageData) return null;

    return (
      <div className="image-preview">
        <img
          src={`data:image/png;base64,${imageData}`}
          alt="Breakpoint input"
        />
      </div>
    );
  };

  const renderGridsCollapsed = (label: string, grids: any) => {
    if (!grids) return null;
    return (
      <div className="collapsed-block">
        <div className="collapsed-header">
          <div className="collapsed-title">{label}</div>
          <button
            className="btn tiny copy-btn"
            type="button"
            onClick={() => {
              void copyJson(grids);
            }}
          >
            Copy JSON
          </button>
        </div>
        <details>
          <summary>Show (collapsed)</summary>
          <pre className="code-block">{JSON.stringify(grids, null, 2)}</pre>
        </details>
      </div>
    );
  };

  const renderCustomForm = () => {
    if (!selectedAgent?.current_breakpoint) return null;
    const point = selectedAgent.current_breakpoint.point;
    const payload = breakpointForm ?? selectedAgent.current_breakpoint.payload ?? {};

    const memoryText = typeof payload.memory_text === "string" ? payload.memory_text : "";
    const memoryLimit =
      typeof payload.memory_word_limit === "number"
        ? payload.memory_word_limit
        : payload.memory_word_limit
          ? Number(payload.memory_word_limit)
          : undefined;

    return (
      <div className="bp-form">
        {renderImagePreview()}
        <div className="bp-form-section">
          <div className="label">Memory</div>
          <textarea
            className="json-editor"
            value={memoryText}
            onChange={(e) => setFormField("memory_text", e.target.value)}
            rows={6}
            placeholder="Scratchpad / memory text"
          />
          <div className="label mt">Memory word limit</div>
          <input
            className="text-input"
            type="number"
            value={memoryLimit ?? ""}
            onChange={(e) => setFormField("memory_word_limit", Number(e.target.value))}
            placeholder="e.g. 500"
          />
        </div>

        {point === "analyze_post" && (
          <div className="bp-form-section">
            <div className="label">Analysis</div>
            <textarea
              className="json-editor"
              value={payload.analysis ?? ""}
              onChange={(e) => setFormField("analysis", e.target.value)}
              rows={10}
            />
          </div>
        )}

        {point === "decide_post" && (
          <div className="bp-form-section">
            <div className="label">Human action</div>
            <textarea
              className="json-editor"
              value={payload.result?.human_action ?? ""}
              onChange={(e) => {
                const next = { ...(payload.result ?? {}) };
                next.human_action = e.target.value;
                setFormField("result", next);
              }}
              rows={4}
            />
            <div className="label mt">Reasoning</div>
            <textarea
              className="json-editor"
              value={payload.result?.reasoning ?? ""}
              onChange={(e) => {
                const next = { ...(payload.result ?? {}) };
                next.reasoning = e.target.value;
                setFormField("result", next);
              }}
              rows={6}
            />
            <div className="label mt">Expected result (optional)</div>
            <textarea
              className="json-editor"
              value={payload.result?.expected_result ?? ""}
              onChange={(e) => {
                const next = { ...(payload.result ?? {}) };
                next.expected_result = e.target.value;
                setFormField("result", next);
              }}
              rows={4}
            />
            <details className="collapsed-block">
              <summary>Advanced: full result JSON</summary>
              <div className="collapsed-header">
                <div className="collapsed-title">Result JSON</div>
                <button
                  className="btn tiny copy-btn"
                  type="button"
                  onClick={() => void copyJson(payload.result ?? {})}
                >
                  Copy JSON
                </button>
              </div>
              <textarea
                className="json-editor"
                value={JSON.stringify(payload.result ?? {}, null, 2)}
                onChange={(e) => {
                  try {
                    setFormField("result", JSON.parse(e.target.value));
                  } catch {
                    // ignore
                  }
                }}
                rows={10}
              />
            </details>
          </div>
        )}

        {point === "convert_post" && (
          <div className="bp-form-section">
            <div className="label">Action (code)</div>
            <input
              className="text-input"
              value={payload.result?.action ?? ""}
              onChange={(e) => {
                const next = { ...(payload.result ?? {}) };
                next.action = e.target.value;
                setFormField("result", next);
              }}
              placeholder="e.g. ACTION6"
            />

            <div className="details-grid mt">
              <div>
                <div className="label">x (only for ACTION6)</div>
                <input
                  className="text-input"
                  type="number"
                  value={payload.result?.x ?? ""}
                  onChange={(e) => {
                    const next = { ...(payload.result ?? {}) };
                    next.x = Number(e.target.value);
                    setFormField("result", next);
                  }}
                />
              </div>
              <div>
                <div className="label">y (only for ACTION6)</div>
                <input
                  className="text-input"
                  type="number"
                  value={payload.result?.y ?? ""}
                  onChange={(e) => {
                    const next = { ...(payload.result ?? {}) };
                    next.y = Number(e.target.value);
                    setFormField("result", next);
                  }}
                />
              </div>
              <div>
                <div className="label">Copy</div>
                <button
                  className="btn tiny copy-btn"
                  type="button"
                  onClick={() => void copyJson(payload.result ?? {})}
                >
                  Copy JSON
                </button>
              </div>
            </div>

            <details className="collapsed-block">
              <summary>Advanced: full action JSON</summary>
              <textarea
                className="json-editor"
                value={JSON.stringify(payload.result ?? {}, null, 2)}
                onChange={(e) => {
                  try {
                    setFormField("result", JSON.parse(e.target.value));
                  } catch {
                    // ignore
                  }
                }}
                rows={12}
              />
            </details>
          </div>
        )}

        {point === "execute_action_pre" && (
          <div className="bp-form-section">
            <div className="label">Action</div>
            <input
              className="text-input"
              value={payload.action ?? ""}
              onChange={(e) => setFormField("action", e.target.value)}
            />
            <div className="label mt">Reasoning</div>
            <textarea
              className="json-editor"
              value={payload.reasoning ?? ""}
              onChange={(e) => setFormField("reasoning", e.target.value)}
              rows={6}
            />
            {renderGridsCollapsed("Action data (JSON)", payload.action_data)}
          </div>
        )}

        {point === "execute_action_post" && (
          <div className="bp-form-section">
            <div className="label">Execute result</div>
            <div className="collapsed-block">
              <div className="collapsed-header">
                <div className="collapsed-title">Result</div>
                <button
                  className="btn tiny copy-btn"
                  type="button"
                  onClick={() => void copyJson(payload.result ?? {})}
                >
                  Copy JSON
                </button>
              </div>
              <details>
                <summary>Show (collapsed)</summary>
                <pre className="code-block">{JSON.stringify(payload.result ?? {}, null, 2)}</pre>
              </details>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderProgressBar = () => {
    if (!selectedAgent) return null;
    // Use last_step from agent broadcasts to show progress as agent moves through steps
    const currentPoint = selectedAgent.last_step || selectedAgent.current_breakpoint?.point;
    if (!currentPoint) return null;
    const idx = BREAKPOINT_POINTS.indexOf(currentPoint as BreakpointPoint);
    return (
      <div className="progress">
        {BREAKPOINT_POINTS.map((p, i) => {
          const state =
            idx === -1 ? "unknown" : i < idx ? "done" : i === idx ? "current" : "todo";
          const label = POINT_LABEL[p];
          return (
            <div key={p} className={`progress-step ${state}`}>
              <div className="progress-label">
                <span className="step-label">{label}</span>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const connectionLabel =
    connectionStatus === "connected"
      ? "Connected"
      : connectionStatus === "connecting"
        ? "Connecting…"
        : connectionStatus === "error"
          ? "Error"
          : "Disconnected";

  return (
    <div className="page">
      <header className="header">
        <div className="header-left">
          <img src="/arcprize.png" alt="ARC Prize" className="header-logo" />
          <div>
          <div className="brand">ARC Breakpoint Tool</div>
          <div className="subtitle">Interactive control over ARC agents</div>
          </div>
        </div>
        <div className="header-right">
          <span className={`status-pill status-${connectionStatus}`}>
            {connectionLabel}
          </span>
          <button className="btn secondary" onClick={handleContinueAll}>
            Continue All
          </button>
        </div>
      </header>

      <main className="layout">
        <section className="sidebar">
          <div className="panel">
            <div className="panel-header">
              <h2>Agents</h2>
              <span className="pill">
                {Object.keys(agents).length} active
              </span>
            </div>
            <div className="panel-body agents-list">
              {Object.values(agents).length === 0 && (
                <div className="empty">No agents connected yet.</div>
              )}
              {Object.values(agents).map((agent) => {
                const isSelected = agent.agent_id === selectedAgentId;
                const isPaused = agent.status === "PAUSED";
                const isDisconnected = agent.status === "DISCONNECTED";
                const canContinue = isPaused && agent.current_breakpoint;
                return (
                  <button
                    key={agent.agent_id}
                    className={`agent-row ${isSelected ? "selected" : ""} ${
                      isPaused ? "paused" : ""
                    }`}
                    onClick={() => setSelectedAgentId(agent.agent_id)}
                  >
                    <div className="agent-row-main">
                      <div className="agent-id">
                        {agent.agent_id.slice(0, 8)}
                      </div>
                      <div className="agent-meta">
                        <div className="agent-config">{agent.config || "—"}</div>
                        <div className="agent-game">
                          {agent.game_id ? `Game · ${agent.game_id}` : "Game · (starting…)"}
                        </div>
                        {agent.last_step && (
                          <div className="agent-step">
                            {formatPointLabel(agent.last_step)}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="agent-row-right">
                      <span
                        className={`status-dot status-${agent.status.toLowerCase()}`}
                      />
                      <span className="score">{agent.score}</span>
                      {isDisconnected ? (
                        <button
                          className="btn tiny remove-btn"
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveAgent(agent.agent_id);
                          }}
                          title="Remove agent"
                        >
                          ×
                        </button>
                      ) : (
                      <button
                        className="btn tiny"
                        type="button"
                          disabled={!canContinue}
                        onClick={(e) => {
                          e.stopPropagation();
                            if (canContinue) {
                          handleContinueAgent(agent.agent_id);
                            }
                        }}
                      >
                        Continue
                      </button>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h3>Global Breakpoints</h3>
              <div className="panel-actions">
                <button
                  className="btn tiny"
                  type="button"
                  onClick={() => handleSetAllGlobalBreakpoints(true)}
                >
                  Set all
                </button>
                <button
                  className="btn tiny"
                  type="button"
                  onClick={() => handleSetAllGlobalBreakpoints(false)}
                >
                  Clear all
                </button>
              </div>
            </div>
            <div className="panel-body">
              {BREAKPOINT_ROWS.map(({ point, label }) => {
                const on = Boolean(globalState.breakpoints[point]);
                return (
                  <div
                    key={point}
                    className={`bp-row bp-clickable ${on ? "active" : ""}`}
                    role="button"
                    tabIndex={0}
                    onClick={() => handleToggleGlobalPoint(point)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") handleToggleGlobalPoint(point);
                    }}
                  >
                    <div className="bp-step">{label}</div>
                    <span className={`bp-status ${on ? "on" : "off"}`}>{on ? "On" : "Off"}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="content">
          {!selectedAgent && (
            <div className="panel full">
              <div className="panel-header">
                <h2>Agent details</h2>
              </div>
              <div className="panel-body">
                <div className="empty">Select an agent to inspect breakpoints.</div>
              </div>
            </div>
          )}

          {selectedAgent && (
            <div className="panel full">
              <div className="panel-header">
                <h2>Agent {selectedAgent.agent_id.slice(0, 8)}</h2>
                <span className="pill muted">
                  {selectedAgent.config || "—"} ·{" "}
                  {selectedAgent.game_id ? `game ${selectedAgent.game_id}` : "game (starting…)"}
                </span>
              </div>
              <div className="panel-body details">
                <div className="details-grid">
                  <div>
                    <div className="label">Status</div>
                    <div className="value">{selectedAgent.status}</div>
                  </div>
                  <div>
                    <div className="label">Score</div>
                    <div className="value">{selectedAgent.score}</div>
                  </div>
                  <div>
                    <div className="label">Last step</div>
                    <div className="value">
                      {selectedAgent.last_step
                        ? formatPointLabel(selectedAgent.last_step)
                        : "—"}
                    </div>
                  </div>
                </div>

                <div className="panel subpanel">
                  <div className="panel-header small">
                    <h3>Agent Breakpoints</h3>
                    <div className="panel-actions">
                      <button
                        className="btn tiny"
                        type="button"
                        onClick={() => handleSetAllAgentBreakpoints(selectedAgent, true)}
                      >
                        Set all
                      </button>
                      <button
                        className="btn tiny"
                        type="button"
                        onClick={() => handleSetAllAgentBreakpoints(selectedAgent, false)}
                      >
                        Clear all
                      </button>
                    </div>
                  </div>
                  <div className="panel-body">
                    {BREAKPOINT_ROWS.map(({ point, label }) => {
                      const on = Boolean(selectedAgent.breakpoints[point]);
                      return (
                        <div
                          key={point}
                          className={`bp-row bp-clickable ${on ? "active" : ""}`}
                          role="button"
                          tabIndex={0}
                          onClick={() => handleToggleAgentPoint(selectedAgent, point)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ")
                              handleToggleAgentPoint(selectedAgent, point);
                          }}
                        >
                          <div className="bp-step">{label}</div>
                          <span className={`bp-status ${on ? "on" : "off"}`}>
                            {on ? "On" : "Off"}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="panel subpanel">
                  <div className="panel-header small">
                    <h3>Current Breakpoint</h3>
                  </div>
                  <div className="panel-body">
                    {!selectedAgent.current_breakpoint && (
                      <div className="empty">
                        This agent is not currently paused at a breakpoint.
                      </div>
                    )}
                    {selectedAgent.current_breakpoint && (
                      <>
                        <div className="details-grid">
                          <div>
                            <div className="label">Current step</div>
                            <div className="value">
                              {selectedAgent.last_step
                                ? formatPointLabel(selectedAgent.last_step)
                                : POINT_LABEL[selectedAgent.current_breakpoint.point]}
                            </div>
                          </div>
                        </div>

                        {renderProgressBar()}

                        {renderCustomForm()}

                        <div className="actions mt">
                          <button
                            className="btn secondary"
                            type="button"
                            onClick={() =>
                              selectedAgent &&
                              handleContinueAgent(selectedAgent.agent_id)
                            }
                          >
                            {breakpointForm ? "Continue and discard changes" : "Continue"}
                          </button>
                          <button
                            className="btn primary"
                            type="button"
                            disabled={!breakpointForm}
                            onClick={handleContinueWithOverrides}
                          >
                            Continue with changes
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};



