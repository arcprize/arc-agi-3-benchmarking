import React, { useEffect, useMemo, useState } from "react";
import type {
  AgentState,
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

export const App: React.FC = () => {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [agents, setAgents] = useState<Record<string, AgentState>>({});
  const [globalState, setGlobalState] = useState<GlobalState>({
    paused: true,
    breakpoints: {
      analyze: true,
      decide: true,
      convert: true,
      execute_action: true,
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
        setAgents((prev) => ({
          ...prev,
          [msg.agent.agent_id]: msg.agent,
        }));
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
        setBreakpointForm(null);
        break;
      }
      case "global_state_updated": {
        setGlobalState(msg.global);
        break;
      }
    }
  };

  const send = (payload: any) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(payload));
  };

  const handleToggleGlobalPaused = () => {
    const next = !globalState.paused;
    send({
      type: "set_global_state",
      global_paused: next,
      breakpoints: globalState.breakpoints,
    });
  };

  const handleToggleGlobalStep = (step: BreakpointStep) => {
    const next = {
      ...globalState.breakpoints,
      [step]: !globalState.breakpoints[step],
    };
    send({
      type: "set_global_state",
      global_paused: globalState.paused,
      breakpoints: next,
    });
  };

  const handleToggleAgentStep = (agent: AgentState, step: BreakpointStep) => {
    const next = {
      ...agent.breakpoints,
      [step]: !agent.breakpoints[step],
    };
    send({
      type: "set_agent_breakpoints",
      agent_id: agent.agent_id,
      breakpoints: next,
    });
  };

  const handleContinueAgent = (agentId: string) => {
    send({ type: "continue_agent", agent_id: agentId });
  };

  const handleContinueAll = () => {
    send({ type: "continue_all" });
  };

  const handleContinueWithOverrides = () => {
    if (!selectedAgent || !selectedAgent.current_breakpoint) return;
    send({
      type: "continue_agent_with_overrides",
      agent_id: selectedAgent.agent_id,
      step: selectedAgent.current_breakpoint.step,
      payload: breakpointForm ?? selectedAgent.current_breakpoint.payload,
    });
  };

  const handleBreakpointJsonChange = (
    e: React.ChangeEvent<HTMLTextAreaElement>,
  ) => {
    const value = e.target.value;
    try {
      const parsed = JSON.parse(value);
      setBreakpointForm(parsed);
    } catch {
      // keep previous valid JSON; user can fix syntax
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !selectedAgent?.current_breakpoint) return;

    const reader = new FileReader();
    reader.onload = () => {
      const base64 = (reader.result as string).split(",")[1];
      const payload = breakpointForm ?? selectedAgent.current_breakpoint!.payload;
      const step = selectedAgent.current_breakpoint!.step;

      const updated = { ...payload };
      if (step === "analyze") {
        if (Array.isArray(updated.current_frame_images) && updated.current_frame_images.length) {
          updated.current_frame_images[0] = {
            kind: "image",
            width: 0,
            height: 0,
            data: base64,
          };
        }
      } else if (step === "decide") {
        if (Array.isArray(updated.frame_images) && updated.frame_images.length) {
          updated.frame_images[0] = {
            kind: "image",
            width: 0,
            height: 0,
            data: base64,
          };
        }
      } else if (step === "convert") {
        if (updated.last_frame_image) {
          updated.last_frame_image = {
            kind: "image",
            width: 0,
            height: 0,
            data: base64,
          };
        }
      }
      setBreakpointForm(updated);
    };
    reader.readAsDataURL(file);
  };

  const renderImagePreview = () => {
    if (!selectedAgent?.current_breakpoint) return null;
    const payload = breakpointForm ?? selectedAgent.current_breakpoint.payload;
    const step = selectedAgent.current_breakpoint.step;

    let imageData: string | null = null;
    if (step === "analyze" && payload.current_frame_images?.[0]?.data) {
      imageData = payload.current_frame_images[0].data;
    } else if (step === "decide" && payload.frame_images?.[0]?.data) {
      imageData = payload.frame_images[0].data;
    } else if (step === "convert" && payload.last_frame_image?.data) {
      imageData = payload.last_frame_image.data;
    }

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
          <div className="brand">ARC Breakpoint Tool</div>
          <div className="subtitle">Interactive control over ARC agents</div>
        </div>
        <div className="header-right">
          <span className={`status-pill status-${connectionStatus}`}>
            {connectionLabel}
          </span>
          <button className="btn secondary" onClick={handleContinueAll}>
            Continue All
          </button>
          <label className="toggle">
            <input
              type="checkbox"
              checked={globalState.paused}
              onChange={handleToggleGlobalPaused}
            />
            <span>Global pause</span>
          </label>
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
                        <span>{agent.config}</span>
                        {agent.game_id && (
                          <span className="muted"> · {agent.game_id}</span>
                        )}
                      </div>
                    </div>
                    <div className="agent-row-right">
                      <span
                        className={`status-dot status-${agent.status.toLowerCase()}`}
                      />
                      <span className="score">{agent.score}</span>
                      <button
                        className="btn tiny"
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleContinueAgent(agent.agent_id);
                        }}
                      >
                        Continue
                      </button>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h3>Global Breakpoints</h3>
            </div>
            <div className="panel-body">
              {BREAKPOINT_STEPS.map((step) => (
                <label key={step} className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={globalState.breakpoints[step]}
                    onChange={() => handleToggleGlobalStep(step)}
                  />
                  <span>{step}</span>
                </label>
              ))}
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
                  {selectedAgent.config} · {selectedAgent.game_id || "no game"}
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
                      {selectedAgent.last_step || "—"}
                    </div>
                  </div>
                </div>

                <div className="panel subpanel">
                  <div className="panel-header small">
                    <h3>Agent Breakpoints</h3>
                  </div>
                  <div className="panel-body">
                    {BREAKPOINT_STEPS.map((step) => (
                      <label key={step} className="checkbox-row">
                        <input
                          type="checkbox"
                          checked={selectedAgent.breakpoints[step]}
                          onChange={() =>
                            handleToggleAgentStep(selectedAgent, step)
                          }
                        />
                        <span>{step}</span>
                      </label>
                    ))}
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
                            <div className="label">Step</div>
                            <div className="value">
                              {selectedAgent.current_breakpoint.step}
                            </div>
                          </div>
                        </div>

                        {renderImagePreview()}

                        <div className="label mt">
                          Payload (JSON – edit to override)
                        </div>
                        <textarea
                          className="json-editor"
                          defaultValue={JSON.stringify(
                            breakpointForm ??
                              selectedAgent.current_breakpoint.payload,
                            null,
                            2,
                          )}
                          onChange={handleBreakpointJsonChange}
                          rows={16}
                        />

                        <div className="label mt">Override image (optional)</div>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleImageUpload}
                        />

                        <div className="actions mt">
                          <button
                            className="btn primary"
                            type="button"
                            onClick={handleContinueWithOverrides}
                          >
                            Continue with changes
                          </button>
                          <button
                            className="btn secondary"
                            type="button"
                            onClick={() =>
                              selectedAgent &&
                              handleContinueAgent(selectedAgent.agent_id)
                            }
                          >
                            Discard changes &amp; continue
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


