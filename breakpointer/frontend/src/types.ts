export type BreakpointStep = "analyze" | "decide" | "convert" | "execute_action";

export interface AgentBreakpoint {
  step: BreakpointStep;
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
  breakpoints: Record<BreakpointStep, boolean>;
}

export interface GlobalState {
  paused: boolean;
  breakpoints: Record<BreakpointStep, boolean>;
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
      payload: any;
    }
  | {
      type: "breakpoint_resolved";
      agent_id: string;
      step: BreakpointStep;
    }
  | {
      type: "global_state_updated";
      global: GlobalState;
    };


