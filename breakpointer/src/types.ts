export interface BreakpointFieldSpec {
  key: string;
  label: string;
  path: string;
  editor: string;
  description?: string;
  read_only?: boolean;
}

export interface BreakpointPointSpec {
  point_id: string;
  label: string;
  phase: string;
  fields: BreakpointFieldSpec[];
  description?: string;
}

export interface BreakpointSectionSpec {
  section_id: string;
  label: string;
  points: BreakpointPointSpec[];
  description?: string;
}

export interface BreakpointSchema {
  sections: BreakpointSectionSpec[];
}

export interface AgentBreakpoint {
  request_id: string;
  point_id: string;
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
  breakpoints: Record<string, boolean>;
  schema?: BreakpointSchema;
}

export interface GlobalState {
  paused: boolean;
  breakpoints: Record<string, boolean>;
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
      point_id: string;
      payload: any;
      request_id: string;
    }
  | {
      type: "breakpoint_resolved";
      agent_id: string;
      point_id: string;
      request_id: string;
    }
  | {
      type: "global_state_updated";
      global: GlobalState;
    };
