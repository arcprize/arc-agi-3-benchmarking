export type BreakpointStep = "analyze" | "decide" | "convert" | "execute_action";
export type BreakpointPhase = "pre" | "post";
// We only support a small set of breakpoint points (intentionally).
export type BreakpointPoint =
  | "analyze_post"
  | "decide_post"
  | "convert_post"
  | "execute_action_pre"
  | "execute_action_post";

export interface AgentBreakpoint {
  request_id: string;
  step: BreakpointStep;
  phase: BreakpointPhase;
  point: BreakpointPoint;
  payload: any;
}

export interface AgentState {
  agent_id: string;
  config?: string;
  card_id?: string;
  game_id?: string;
  status: "IDLE" | "CONNECTED" | "RUNNING" | "PAUSED" | "DISCONNECTED";
  score: number;
  last_step?: string;
  current_breakpoint?: AgentBreakpoint | null;
  breakpoints: Record<BreakpointPoint, boolean>;
  play_num?: number;
  play_action_counter?: number;
  action_counter?: number;
}

export interface GlobalState {
  breakpoints: Record<BreakpointPoint, boolean>;
}

export interface ServerStateSnapshotMessage {
  type: "state_snapshot";
  global: GlobalState;
  agents: AgentState[];
}

export type ServerMessage =
  | ServerStateSnapshotMessage
  | {
      type: "agent_updated";
      agent: AgentState;
    }
  | {
      type: "breakpoint_pending";
      agent: AgentState;
      step: BreakpointStep;
      phase: BreakpointPhase;
      point: BreakpointPoint;
      request_id: string;
      payload: any;
    }
  | {
      type: "breakpoint_resolved";
      agent_id: string;
      step: BreakpointStep;
      phase: BreakpointPhase;
      point: BreakpointPoint;
      request_id: string;
    }
  | {
      type: "global_state_updated";
      global: GlobalState;
    }
  | {
      type: "agent_removed";
      agent_id: string;
    };


