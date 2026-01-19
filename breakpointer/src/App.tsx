import React, { useEffect, useMemo, useState } from "react";
import type {
  AgentState,
  GlobalState,
  ServerMessage,
  BreakpointPointSpec,
  BreakpointSectionSpec,
} from "./types";

type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

// Helper to format point labels
const formatPointLabel = (pointId: string): string => {
  // Convert point_id like "analyze.post" to "Analyze (Post)"
  const parts = pointId.split(".");
  if (parts.length === 2) {
    const [section, phase] = parts;
    const sectionLabel = section.charAt(0).toUpperCase() + section.slice(1).replace(/_/g, " ");
    const phaseLabel = phase.charAt(0).toUpperCase() + phase.slice(1);
    return `${sectionLabel} (${phaseLabel})`;
  }
  return pointId.toUpperCase().replace(/_/g, " ");
};

// Helper to find image in payload
const findImage = (payload: any): string | null => {
  if (!payload || typeof payload !== "object") return null;
  const candidates = [
    payload.latest_frame_image,
    payload.current_frame_image,
    payload.previous_frame_image,
    payload.current_frame_images?.[0],
    payload.frame_images?.[0],
    payload.last_frame_image,
  ];
  for (const candidate of candidates) {
    if (candidate?.kind === "image" && candidate?.data) {
      return candidate.data as string;
    }
  }
  return null;
};

// Helper to find point in schema
const findPoint = (
  agent: AgentState | null,
  pointId: string | undefined,
): BreakpointPointSpec | undefined => {
  if (!agent?.schema || !pointId) return undefined;
  for (const section of agent.schema.sections || []) {
    for (const point of section.points || []) {
      if (point.point_id === pointId) return point;
    }
  }
  return undefined;
};

// Get all point IDs from an agent's schema
const getAllPointIds = (agent: AgentState | null): string[] => {
  if (!agent?.schema) return [];
  const points: string[] = [];
  for (const section of agent.schema.sections || []) {
    for (const point of section.points || []) {
      points.push(point.point_id);
    }
  }
  return points;
};

export const App: React.FC = () => {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [agents, setAgents] = useState<Record<string, AgentState>>({});
  const [globalState, setGlobalState] = useState<GlobalState>({
    paused: false,
    breakpoints: {},
  });
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [breakpointForm, setBreakpointForm] = useState<any | null>(null);
  const [modalImage, setModalImage] = useState<{ src: string; alt: string } | null>(null);

  const selectedAgent: AgentState | null = useMemo(() => {
    if (!selectedAgentId) return null;
    return agents[selectedAgentId] || null;
  }, [selectedAgentId, agents]);

  useEffect(() => {
    const url = `ws://${window.location.hostname}:8765/ws`;
    const socket = new WebSocket(url);
    setConnectionStatus("connecting");

    socket.onopen = () => {
      setConnectionStatus("connected");
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

  useEffect(() => {
    const payload = selectedAgent?.current_breakpoint?.payload ?? null;
    setBreakpointForm(payload);
  }, [
    selectedAgent?.agent_id,
    selectedAgent?.current_breakpoint?.request_id,
  ]);

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
          if (msg.agent.agent_id in prev || msg.agent.status !== "DISCONNECTED") {
            return {
              ...prev,
              [msg.agent.agent_id]: msg.agent,
            };
          }
          return prev;
        });
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
        if (selectedAgentId === msg.agent.agent_id) {
          setBreakpointForm(msg.payload);
        }
        break;
      }
      case "breakpoint_resolved": {
        if (selectedAgentId === msg.agent_id) {
          setBreakpointForm(null);
        }
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

  const handleToggleGlobalPoint = (pointId: string) => {
    const next = {
      ...globalState.breakpoints,
      [pointId]: !globalState.breakpoints[pointId],
    };
    send({
      type: "set_global_state",
      paused: globalState.paused,
      breakpoints: next,
    });
  };

  const handleSetAllGlobalBreakpoints = (value: boolean) => {
    const allPointIds = new Set<string>();
    Object.values(agents).forEach((agent) => {
      getAllPointIds(agent).forEach((pid) => allPointIds.add(pid));
    });
    const next: Record<string, boolean> = {};
    allPointIds.forEach((pid) => {
      next[pid] = value;
    });
    send({
      type: "set_global_state",
      paused: globalState.paused,
      breakpoints: next,
    });
  };

  const handleToggleAgentPoint = (agent: AgentState, pointId: string) => {
    const next = {
      ...agent.breakpoints,
      [pointId]: !agent.breakpoints[pointId],
    };
    send({
      type: "set_agent_breakpoints",
      agent_id: agent.agent_id,
      breakpoints: next,
    });
  };

  const handleSetAllAgentBreakpoints = (agent: AgentState, value: boolean) => {
    const pointIds = getAllPointIds(agent);
    const next: Record<string, boolean> = {};
    pointIds.forEach((pid) => {
      next[pid] = value;
    });
    send({
      type: "set_agent_breakpoints",
      agent_id: agent.agent_id,
      breakpoints: next,
    });
  };

  const handleContinueAgent = (agentId: string) => {
    const agent = agents[agentId];
    if (
      agent &&
      agent.status === "PAUSED" &&
      agent.current_breakpoint &&
      agent.current_breakpoint.request_id
    ) {
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

    const previousImageData: string | null = payload.previous_frame_image?.data ?? null;
    const helperImageData: string | null = payload.helper_image?.data ?? null;
    const currentImageData: string | null = payload.current_frame_image?.data ?? payload.latest_frame_image?.data ?? null;

    if (!previousImageData && !helperImageData && !currentImageData) return null;

    const handleImageClick = (imageData: string, alt: string) => {
      if (modalImage && modalImage.src === `data:image/png;base64,${imageData}`) {
        setModalImage(null);
      } else {
        setModalImage({ src: `data:image/png;base64,${imageData}`, alt });
      }
    };

    return (
      <>
        <div className="image-preview" style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          {previousImageData && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div style={{ fontSize: '12px', marginBottom: '4px', color: '#666' }}>Prior</div>
              <img
                src={`data:image/png;base64,${previousImageData}`}
                alt="Prior frame"
                onClick={() => handleImageClick(previousImageData, "Prior frame")}
                style={{
                  maxWidth: '350px',
                  height: 'auto',
                  border: '1px solid #ddd',
                  cursor: 'pointer',
                  transition: 'opacity 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
                onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
              />
            </div>
          )}
          {helperImageData && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div style={{ fontSize: '12px', marginBottom: '4px', color: '#666' }}>Helper</div>
              <img
                src={`data:image/png;base64,${helperImageData}`}
                alt="Helper diff image"
                onClick={() => handleImageClick(helperImageData, "Helper diff image")}
                style={{
                  maxWidth: '350px',
                  height: 'auto',
                  border: '1px solid #ddd',
                  cursor: 'pointer',
                  transition: 'opacity 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
                onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
              />
            </div>
          )}
          {currentImageData && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div style={{ fontSize: '12px', marginBottom: '4px', color: '#666' }}>Current</div>
              <img
                src={`data:image/png;base64,${currentImageData}`}
                alt="Current frame"
                onClick={() => handleImageClick(currentImageData, "Current frame")}
                style={{
                  maxWidth: '350px',
                  height: 'auto',
                  border: '1px solid #ddd',
                  cursor: 'pointer',
                  transition: 'opacity 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
                onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
              />
            </div>
          )}
        </div>
        {modalImage && (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.9)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 10000,
              cursor: 'pointer'
            }}
            onClick={() => setModalImage(null)}
          >
            <img
              src={modalImage.src}
              alt={modalImage.alt}
              style={{
                maxWidth: '90vw',
                maxHeight: '90vh',
                objectFit: 'contain',
                border: '2px solid #fff'
              }}
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        )}
      </>
    );
  };

  // Helper to get value from payload using JSON path
  const getValueByPath = (obj: any, path: string): any => {
    const parts = path.split(".");
    let current = obj;
    for (const part of parts) {
      if (current == null || typeof current !== "object") return undefined;
      current = current[part];
    }
    return current;
  };

  // Helper to set value in payload using JSON path
  const setValueByPath = (obj: any, path: string, value: any): any => {
    const parts = path.split(".");
    const result = JSON.parse(JSON.stringify(obj)); // Deep clone
    let current = result;
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      if (current[part] == null || typeof current[part] !== "object") {
        current[part] = {};
      }
      current = current[part];
    }
    current[parts[parts.length - 1]] = value;
    return result;
  };

  const renderCustomForm = () => {
    if (!selectedAgent?.current_breakpoint) return null;
    const pointId = selectedAgent.current_breakpoint.point_id;
    const payload = breakpointForm ?? selectedAgent.current_breakpoint.payload ?? {};

    // Find the point spec from the schema
    const activePoint = findPoint(selectedAgent, pointId);
    const fields = activePoint?.fields || [];

    return (
      <div className="bp-form">
        {renderImagePreview()}

        {fields.length > 0 ? (
          // Render fields from schema
          fields.map((field) => {
            const value = getValueByPath(payload, field.path);
            const isReadOnly = field.read_only || false;

            return (
              <div key={field.key} className="bp-form-section">
                <div className="label">
                  {field.label}
                  {field.description && (
                    <span className="hint" style={{ display: "block", marginTop: "4px", fontSize: "13px" }}>
                      {field.description}
                    </span>
                  )}
                </div>
                {field.editor === "text" && (
                  <input
                    className="text-input"
                    type="text"
                    value={value ?? ""}
                    onChange={(e) => {
                      const updated = setValueByPath(payload, field.path, e.target.value);
                      setBreakpointForm(updated);
                    }}
                    disabled={isReadOnly}
                    placeholder={field.description || ""}
                  />
                )}
                {field.editor === "number" && (
                  <input
                    className="text-input"
                    type="number"
                    value={value ?? ""}
                    onChange={(e) => {
                      const numValue = e.target.value === "" ? undefined : Number(e.target.value);
                      const updated = setValueByPath(payload, field.path, numValue);
                      setBreakpointForm(updated);
                    }}
                    disabled={isReadOnly}
                    placeholder={field.description || ""}
                  />
                )}
                {field.editor === "textarea" && (
                  <textarea
                    className="json-editor"
                    value={typeof value === "string" ? value : value != null ? String(value) : ""}
                    onChange={(e) => {
                      const updated = setValueByPath(payload, field.path, e.target.value);
                      setBreakpointForm(updated);
                    }}
                    disabled={isReadOnly}
                    rows={6}
                    placeholder={field.description || ""}
                  />
                )}
                {field.editor === "json" && (
                  <textarea
                    className="json-editor"
                    value={
                      value != null
                        ? typeof value === "string"
                          ? value
                          : JSON.stringify(value, null, 2)
                        : ""
                    }
                    onChange={(e) => {
                      try {
                        const parsed = e.target.value.trim() === "" ? null : JSON.parse(e.target.value);
                        const updated = setValueByPath(payload, field.path, parsed);
                        setBreakpointForm(updated);
                      } catch {
                        // Invalid JSON, but still update the form so user can continue editing
                        const updated = setValueByPath(payload, field.path, e.target.value);
                        setBreakpointForm(updated);
                      }
                    }}
                    disabled={isReadOnly}
                    rows={10}
                    placeholder={field.description || "{}"}
                  />
                )}
                {field.editor === "boolean" && (
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={Boolean(value)}
                      onChange={(e) => {
                        const updated = setValueByPath(payload, field.path, e.target.checked);
                        setBreakpointForm(updated);
                      }}
                      disabled={isReadOnly}
                    />
                    <span>{value ? "True" : "False"}</span>
                  </label>
                )}
              </div>
            );
          })
        ) : (
          // Fallback: show full JSON editor if no fields defined
          <div className="bp-form-section">
            <div className="label">Payload (JSON – edit to override)</div>
            <textarea
              className="json-editor"
              value={JSON.stringify(payload, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = JSON.parse(e.target.value);
                  setBreakpointForm(parsed);
                } catch {
                  // ignore invalid JSON
                }
              }}
              rows={12}
            />
          </div>
        )}

        {/* Always show full JSON editor as advanced option */}
        <details className="collapsed-block">
          <summary>Advanced: Full Payload (JSON)</summary>
          <div className="collapsed-header">
            <div className="collapsed-title">Full Payload</div>
            <button
              className="btn tiny copy-btn"
              type="button"
              onClick={() => void copyJson(payload)}
            >
              Copy JSON
            </button>
          </div>
          <textarea
            className="json-editor"
            value={JSON.stringify(payload, null, 2)}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                setBreakpointForm(parsed);
              } catch {
                // ignore
              }
            }}
            rows={12}
          />
        </details>
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

  const globalPointIds = Object.keys(globalState.breakpoints || {});

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
                    onClick={() => setSelectedAgentId(isSelected ? null : agent.agent_id)}
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
                        {(typeof agent.play_num === "number" || typeof agent.play_action_counter === "number") && (
                          <div className="agent-step">
                            {typeof agent.play_num === "number" && `Play ${agent.play_num}`}
                            {typeof agent.play_num === "number" && typeof agent.play_action_counter === "number" && " · "}
                            {typeof agent.play_action_counter === "number" && `Step ${agent.play_action_counter}`}
                          </div>
                        )}
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

          {globalPointIds.length > 0 && (
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
                {globalPointIds.map((pointId) => {
                  const on = Boolean(globalState.breakpoints[pointId]);
                  return (
                    <div
                      key={pointId}
                      className={`bp-row bp-clickable ${on ? "active" : ""}`}
                      role="button"
                      tabIndex={0}
                      onClick={() => handleToggleGlobalPoint(pointId)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") handleToggleGlobalPoint(pointId);
                      }}
                    >
                      <div className="bp-step">{formatPointLabel(pointId)}</div>
                      <span className={`bp-status ${on ? "on" : "off"}`}>{on ? "On" : "Off"}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
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
                  {typeof selectedAgent.play_num === "number" && ` · Play ${selectedAgent.play_num}`}
                  {typeof selectedAgent.play_action_counter === "number" && ` · Step ${selectedAgent.play_action_counter}`}
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
                    {selectedAgent.schema?.sections?.length ? (
                      selectedAgent.schema.sections.map((section) => (
                        <div key={section.section_id} className="section-block">
                          <div className="label" style={{ marginBottom: "12px", fontSize: "16px" }}>
                            {section.label}
                          </div>
                          {section.points.map((point) => {
                            const on = Boolean(selectedAgent.breakpoints?.[point.point_id]);
                            return (
                              <div
                                key={point.point_id}
                                className={`bp-row bp-clickable ${on ? "active" : ""}`}
                                role="button"
                                tabIndex={0}
                                onClick={() => handleToggleAgentPoint(selectedAgent, point.point_id)}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter" || e.key === " ")
                                    handleToggleAgentPoint(selectedAgent, point.point_id);
                                }}
                              >
                                <div className="bp-step">{point.label}</div>
                                <span className={`bp-status ${on ? "on" : "off"}`}>
                                  {on ? "On" : "Off"}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      ))
                    ) : (
                      <div className="empty">
                        No schema reported for this agent.
                      </div>
                    )}
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
                                : formatPointLabel(selectedAgent.current_breakpoint.point_id)}
                            </div>
                          </div>
                        </div>

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
