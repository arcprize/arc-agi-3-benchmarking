import asyncio
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer
from typing import Any, Dict, List, Optional, Tuple

import websockets
from websockets.server import WebSocketServerProtocol


logger = logging.getLogger(__name__)


class StaticFileHandler(SimpleHTTPRequestHandler):
    """Serve static files for the React breakpoint UI."""

    def __init__(self, *args, directory: Optional[str] = None, **kwargs):
        if directory is None:
            directory = getattr(self, "directory", os.getcwd())
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        # Local dev tool: disable caching so UI changes show up immediately
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def start_http_server(port: int, static_dir: str) -> ThreadingTCPServer:
    """Start a simple threaded HTTP server serving files from `static_dir`."""
    handler_class = StaticFileHandler
    handler_class.directory = static_dir

    httpd = ThreadingTCPServer(("0.0.0.0", port), handler_class)
    thread = threading.Thread(
        target=httpd.serve_forever,
        name=f"BreakpointHTTPServer:{port}",
        daemon=True,
    )
    thread.start()
    logger.info("HTTP static server serving %s on http://localhost:%d", static_dir, port)
    return httpd


@dataclass
class AgentState:
    agent_id: str
    config: Optional[str] = None
    card_id: Optional[str] = None
    game_id: Optional[str] = None
    status: str = "IDLE"
    score: int = 0
    last_step: Optional[str] = None
    current_breakpoint: Optional[Dict[str, Any]] = None
    breakpoints: Dict[str, bool] = field(
        default_factory=lambda: {
            "analyze_post": True,
            "decide_post": True,
            "convert_post": True,
            "execute_action_pre": True,
            "execute_action_post": True,
        }
    )
    last_heartbeat: float = 0.0
    ws: Optional[WebSocketServerProtocol] = None
    play_num: Optional[int] = None
    play_action_counter: Optional[int] = None
    action_counter: Optional[int] = None


class BreakpointServerState:
    """In-memory state shared between agents and UI clients."""

    def __init__(self) -> None:
        self.global_breakpoints: Dict[str, bool] = {
            "analyze_post": True,
            "decide_post": True,
            "convert_post": True,
            "execute_action_pre": True,
            "execute_action_post": True,
        }
        self.agents: Dict[str, AgentState] = {}
        self.ui_clients: List[WebSocketServerProtocol] = []
        # Pending breakpoint resolutions, keyed by request_id
        self.pending: Dict[str, asyncio.Future] = {}
        # Resolved breakpoint continuations that couldn't be delivered immediately,
        # keyed by agent_id then request_id.
        self.pending_continues: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._heartbeat_timeout: float = 10.0  # seconds

    @staticmethod
    def _req_step_key(agent_id: str, request_id: str) -> str:
        return f"{agent_id}:{request_id}"


class BreakpointWebSocketServer:
    """Websocket server handling agent + UI messages for breakpoints."""

    def __init__(self, host: str, port: int, state: BreakpointServerState) -> None:
        self._host = host
        self._port = port
        self._state = state
        self._server: Optional[websockets.server.Serve] = None
        # Background tasks waiting on breakpoint futures (key = "{agent_id}:{request_id}")
        self._pending_tasks: Dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        # Configure websocket server with ping/pong keepalive to prevent premature disconnections
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
            ping_interval=20,  # Send ping every 20 seconds
            ping_timeout=10,   # Wait 10 seconds for pong response
            close_timeout=10,  # Wait 10 seconds for close handshake
        )
        logger.info(
            "Breakpoint WebSocket server listening on ws://%s:%d", self._host, self._port
        )

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        client_kind: Optional[str] = None
        client_agent_id: Optional[str] = None

        try:
            raw = await ws.recv()
            msg = json.loads(raw)
            client_kind = msg.get("client", "ui")

            if client_kind == "ui":
                await self._register_ui(ws)
                await self._send_full_state(ws)
            elif client_kind == "agent":
                client_agent_id = msg.get("agent_id")
                if not client_agent_id:
                    raise ValueError("agent client must send agent_id")
                await self._handle_agent_hello(ws, msg)
            else:
                raise ValueError(f"Unknown client type: {client_kind}")

            async for raw_msg in ws:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message, ignoring")
                    continue

                if client_kind == "ui":
                    await self._handle_ui_message(ws, data)
                elif client_kind == "agent":
                    await self._handle_agent_message(ws, data)
        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError) as e:
            if client_kind == "agent" and client_agent_id:
                logger.info(
                    "Agent %s WebSocket connection closed: %s (code=%s, reason=%s)",
                    client_agent_id,
                    type(e).__name__,
                    getattr(e, "code", None),
                    getattr(e, "reason", None),
                )
            else:
                logger.info("WebSocket connection closed: %s", type(e).__name__)
        except Exception as exc:
            logger.error("WebSocket handler error: %s", exc, exc_info=True)
        finally:
            if client_kind == "ui":
                await self._unregister_ui(ws)
            elif client_kind == "agent" and client_agent_id:
                # Only mark as disconnected if this websocket is still the active one
                # This prevents race conditions where agent reconnects quickly
                state = self._state.agents.get(client_agent_id)
                if state and state.ws == ws:
                    await self._handle_agent_disconnected(client_agent_id)

    # UI helpers --------------------------------------------------------

    async def _register_ui(self, ws: WebSocketServerProtocol) -> None:
        self._state.ui_clients.append(ws)
        logger.info("UI client connected")

    async def _unregister_ui(self, ws: WebSocketServerProtocol) -> None:
        if ws in self._state.ui_clients:
            self._state.ui_clients.remove(ws)
            logger.info("UI client disconnected")

    async def _send_full_state(self, ws: WebSocketServerProtocol) -> None:
        payload = {
            "type": "state_snapshot",
            "global": {
                "breakpoints": self._state.global_breakpoints,
            },
            "agents": [self._agent_to_dict(a) for a in self._state.agents.values()],
        }
        await ws.send(json.dumps(payload))

    async def _broadcast(self, message: Dict[str, Any]) -> None:
        if not self._state.ui_clients:
            return
        encoded = json.dumps(message)
        await asyncio.gather(
            *[self._safe_send(ws, encoded) for ws in list(self._state.ui_clients)],
            return_exceptions=True,
        )

    async def _safe_send(self, ws: WebSocketServerProtocol, data: str) -> None:
        try:
            await ws.send(data)
        except Exception:
            logger.debug("Failed to send to UI client", exc_info=True)

    async def _handle_ui_message(
        self, ws: WebSocketServerProtocol, data: Dict[str, Any]
    ) -> None:
        msg_type = data.get("type")

        if msg_type == "set_global_state":
            breakpoints = data.get("breakpoints", self._state.global_breakpoints)
            self._state.global_breakpoints.update({k: bool(v) for k, v in breakpoints.items()})
            await self._broadcast(
                {
                    "type": "global_state_updated",
                    "global": {
                        "breakpoints": self._state.global_breakpoints,
                    },
                }
            )

        elif msg_type == "set_agent_breakpoints":
            agent_id = data.get("agent_id")
            bp = data.get("breakpoints", {})
            if agent_id and agent_id in self._state.agents:
                self._state.agents[agent_id].breakpoints.update(
                    {k: bool(v) for k, v in bp.items()}
                )
                await self._broadcast(
                    {
                        "type": "agent_updated",
                        "agent": self._agent_to_dict(self._state.agents[agent_id]),
                    }
                )

        elif msg_type == "continue_request":
            agent_id = data.get("agent_id")
            request_id = data.get("request_id")
            payload = data.get("payload") if "payload" in data else None
            if agent_id and request_id:
                await self._resolve_pending(
                    agent_id=agent_id, request_id=request_id, payload=payload
                )

        elif msg_type == "continue_all":
            # Resolve all pending requests with no overrides
            for agent in list(self._state.agents.values()):
                if agent.current_breakpoint and agent.status == "PAUSED":
                    req_id = agent.current_breakpoint.get("request_id")
                    if req_id:
                        await self._resolve_pending(
                            agent_id=agent.agent_id, request_id=req_id, payload=None
                        )

        elif msg_type == "remove_agent":
            agent_id = data.get("agent_id")
            if agent_id and agent_id in self._state.agents:
                # Cancel any pending breakpoint background tasks
                task_keys_to_remove = [
                    key
                    for key in list(self._pending_tasks.keys())
                    if key.startswith(f"{agent_id}:")
                ]
                for key in task_keys_to_remove:
                    task = self._pending_tasks.pop(key, None)
                    if task and not task.done():
                        task.cancel()
                # Cancel any pending breakpoints
                # Note: pending is keyed by request_id; cancel current request if present
                state = self._state.agents.get(agent_id)
                if state and state.current_breakpoint:
                    req_id = state.current_breakpoint.get("request_id")
                    if req_id:
                        fut = self._state.pending.pop(req_id, None)
                        if fut and not fut.done():
                            fut.cancel()
                self._state.pending_continues.pop(agent_id, None)
                # Remove agent from state
                del self._state.agents[agent_id]
                # Broadcast removal to UI
                await self._broadcast(
                    {"type": "agent_removed", "agent_id": agent_id}
                )

    # Agent helpers -----------------------------------------------------

    async def _handle_agent_hello(
        self, ws: WebSocketServerProtocol, msg: Dict[str, Any]
    ) -> None:
        agent_id = msg.get("agent_id")
        if not agent_id:
            raise ValueError("agent_hello requires agent_id")

        state = self._state.agents.get(agent_id)
        if not state:
            state = AgentState(agent_id=agent_id)
            self._state.agents[agent_id] = state

        state.config = msg.get("config", state.config)
        state.card_id = msg.get("card_id", state.card_id)
        state.game_id = msg.get("game_id", state.game_id)
        state.status = "CONNECTED"
        state.ws = ws
        state.last_heartbeat = time.time()
        # Ensure breakpoint dictionaries contain all current keys (handles older agents lingering in memory)
        for key, default_val in self._state.global_breakpoints.items():
            if key not in state.breakpoints:
                state.breakpoints[key] = bool(default_val)

        await self._broadcast(
            {"type": "agent_updated", "agent": self._agent_to_dict(state)}
        )

        await ws.send(
            json.dumps(
                {
                    "type": "hello_ack",
                    "global": {
                        "breakpoints": self._state.global_breakpoints,
                    },
                }
            )
        )

        # If we have any pending continues queued for this agent, attempt delivery now.
        queued = self._state.pending_continues.get(agent_id) or {}
        if queued:
            # Deliver in insertion order; after successful send, remove.
            for req_id, payload in list(queued.items()):
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "breakpoint_continue",
                                "request_id": req_id,
                                "payload": payload,
                            }
                        )
                    )
                    queued.pop(req_id, None)
                except Exception:
                    logger.info(
                        "Failed to deliver queued breakpoint_continue to agent %s (request_id=%s)",
                        agent_id,
                        req_id,
                        exc_info=True,
                    )
                    break
            if not queued:
                self._state.pending_continues.pop(agent_id, None)

    async def _handle_agent_message(
        self, ws: WebSocketServerProtocol, data: Dict[str, Any]
    ) -> None:
        msg_type = data.get("type")
        agent_id = data.get("agent_id")
        if not agent_id:
            return

        if msg_type == "agent_update":
            state = self._state.agents.get(agent_id)
            if not state:
                state = AgentState(agent_id=agent_id)
                self._state.agents[agent_id] = state
            # Update websocket reference and heartbeat to ensure we're tracking the correct connection
            state.ws = ws
            state.last_heartbeat = time.time()
            # Allow agent to refresh identity fields (useful when game_id becomes available after play_game starts)
            if "config" in data and data.get("config") is not None:
                state.config = data.get("config")
            if "card_id" in data and data.get("card_id") is not None:
                state.card_id = data.get("card_id")
            if "game_id" in data and data.get("game_id") is not None:
                state.game_id = data.get("game_id")
            state.status = data.get("status", state.status)
            state.score = int(data.get("score", state.score))
            state.last_step = data.get("step_name", state.last_step)
            if "play_num" in data:
                state.play_num = data.get("play_num")
            if "play_action_counter" in data:
                state.play_action_counter = data.get("play_action_counter")
            if "action_counter" in data:
                state.action_counter = data.get("action_counter")
            await self._broadcast(
                {"type": "agent_updated", "agent": self._agent_to_dict(state)}
            )

        elif msg_type == "breakpoint_request":
            step = data.get("step")
            phase = data.get("phase")
            point = data.get("point")
            payload = data.get("payload") or {}
            request_id = data.get("request_id")
            if not step or phase not in ("pre", "post") or not point or not request_id:
                logger.warning("Invalid breakpoint_request (missing fields): %s", data)
                return

            state = self._state.agents.get(agent_id)
            if not state:
                state = AgentState(agent_id=agent_id)
                self._state.agents[agent_id] = state

            # Update websocket reference to ensure we're using the current connection
            state.ws = ws
            state.last_heartbeat = time.time()
            state.last_step = step

            # Pause if EITHER global OR agent has this point enabled.
            should_pause = bool(self._state.global_breakpoints.get(point, False)) or bool(
                state.breakpoints.get(point, False)
            )

            if not should_pause:
                # Auto-continue: respond immediately without involving UI.
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "breakpoint_continue",
                                "request_id": request_id,
                                "step": step,
                                "payload": payload,
                            }
                        )
                    )
                except Exception:
                    logger.info(
                        "Failed to auto-continue breakpoint for agent %s (step=%s)",
                        agent_id,
                        step,
                        exc_info=True,
                    )
                return

            state.status = "PAUSED"
            state.current_breakpoint = {
                "request_id": request_id,
                "step": step,
                "phase": phase,
                "point": point,
                "payload": payload,
            }

            await self._broadcast(
                {
                    "type": "breakpoint_pending",
                    "agent": self._agent_to_dict(state),
                    "step": step,
                    "phase": phase,
                    "point": point,
                    "request_id": request_id,
                    "payload": payload,
                }
            )

            # Store pending future keyed by request_id
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._state.pending[request_id] = fut
            # IMPORTANT: don't await the future here. If we block inside the agent socket
            # handler, we stop reading heartbeats and the server will incorrectly time out
            # the agent while it's paused at a breakpoint. Instead, wait in a background task.
            task_key = self._state._req_step_key(agent_id, request_id)
            old_task = self._pending_tasks.pop(task_key, None)
            if old_task and not old_task.done():
                old_task.cancel()
            self._pending_tasks[task_key] = asyncio.create_task(
                self._breakpoint_wait_and_continue(
                    agent_id=agent_id,
                    step=step,
                    phase=phase,
                    point=point,
                    request_id=request_id,
                    original_payload=payload,
                    fut=fut,
                )
            )

        elif msg_type == "agent_disconnected":
            await self._handle_agent_disconnected(agent_id)

        elif msg_type == "heartbeat":
            state = self._state.agents.get(agent_id)
            if state:
                # Always update websocket reference and heartbeat time
                # This ensures we're tracking the correct connection even if agent reconnected
                state.ws = ws
                state.last_heartbeat = time.time()
                # Update status if it was disconnected but now reconnected
                if state.status == "DISCONNECTED":
                    state.status = "CONNECTED"
                    await self._broadcast(
                        {"type": "agent_updated", "agent": self._agent_to_dict(state)}
                    )

    async def _handle_agent_disconnected(self, agent_id: str) -> None:
        state = self._state.agents.get(agent_id)
        if not state:
            return
        old_status = state.status
        state.status = "DISCONNECTED"
        state.current_breakpoint = None
        state.ws = None
        # Cancel any pending breakpoint background tasks for this agent
        task_keys_to_remove = [
            key for key in list(self._pending_tasks.keys()) if key.startswith(f"{agent_id}:")
        ]
        for key in task_keys_to_remove:
            task = self._pending_tasks.pop(key, None)
            if task and not task.done():
                task.cancel()
        # Cancel any pending breakpoints for this agent
        if state.current_breakpoint:
            req_id = state.current_breakpoint.get("request_id")
            if req_id:
                fut = self._state.pending.pop(req_id, None)
                if fut and not fut.done():
                    fut.cancel()
        if old_status != "DISCONNECTED":
            await self._broadcast(
                {"type": "agent_updated", "agent": self._agent_to_dict(state)}
            )
            logger.info("Agent %s marked as DISCONNECTED", agent_id)

    async def _breakpoint_wait_and_continue(
        self,
        agent_id: str,
        step: str,
        phase: str,
        point: str,
        request_id: str,
        original_payload: Dict[str, Any],
        fut: asyncio.Future,
    ) -> None:
        """Wait for UI to resolve a breakpoint, then send continue back to the agent."""
        task_key = self._state._req_step_key(agent_id, request_id)
        try:
            overrides = await fut

            # Clean up server-side state
            state = self._state.agents.get(agent_id)
            if state:
                state.status = "RUNNING"
                state.current_breakpoint = None
                await self._broadcast(
                    {"type": "agent_updated", "agent": self._agent_to_dict(state)}
                )

            await self._broadcast(
                {
                    "type": "breakpoint_resolved",
                    "agent_id": agent_id,
                    "step": step,
                    "phase": phase,
                    "point": point,
                    "request_id": request_id,
                }
            )

            payload_to_send = overrides if overrides is not None else original_payload
            # Prefer current agent websocket (handles reconnects)
            ws_to_use: Optional[WebSocketServerProtocol] = None
            if state:
                ws_to_use = state.ws
            if ws_to_use is None:
                # Agent isn't connected; queue for later delivery.
                self._state.pending_continues.setdefault(agent_id, {})[request_id] = payload_to_send
                return

            try:
                await ws_to_use.send(
                    json.dumps(
                        {
                            "type": "breakpoint_continue",
                            "request_id": request_id,
                            "step": step,
                            "phase": phase,
                            "point": point,
                            "payload": payload_to_send,
                        }
                    )
                )
            except Exception:
                logger.info(
                    "Failed to send breakpoint_continue to agent %s (step=%s, request_id=%s) - queued",
                    agent_id,
                    step,
                    request_id,
                    exc_info=True,
                )
                self._state.pending_continues.setdefault(agent_id, {})[request_id] = payload_to_send
        except asyncio.CancelledError:
            # Agent disconnected or was replaced; nothing to do.
            pass
        except Exception:
            logger.error(
                "Error while waiting for breakpoint resolution (agent=%s, step=%s)",
                agent_id,
                step,
                exc_info=True,
            )
        finally:
            # Always remove pending bookkeeping if it's still present.
            self._state.pending.pop(request_id, None)
            current = self._pending_tasks.get(task_key)
            if current is not None and current is asyncio.current_task():
                self._pending_tasks.pop(task_key, None)

    async def _resolve_pending(
        self,
        agent_id: str,
        request_id: Optional[str],
        payload: Optional[Dict[str, Any]],
    ) -> None:
        """
        Resolve a pending breakpoint future.

        - If request_id is provided, resolve that request.
        - Otherwise, resolve the currently paused breakpoint for the agent (if any).
        """
        state = self._state.agents.get(agent_id)
        if not state or not state.current_breakpoint:
            return

        req_id = request_id or state.current_breakpoint.get("request_id")
        if not req_id:
            return

        fut = self._state.pending.get(req_id)
        if fut and not fut.done():
            fut.set_result(payload)

    @staticmethod
    def _agent_to_dict(agent: AgentState) -> Dict[str, Any]:
        return {
            "agent_id": agent.agent_id,
            "config": agent.config,
            "card_id": agent.card_id,
            "game_id": agent.game_id,
            "status": agent.status,
            "score": agent.score,
            "last_step": agent.last_step,
            "current_breakpoint": agent.current_breakpoint,
            "breakpoints": agent.breakpoints,
            "play_num": getattr(agent, "play_num", None),
            "play_action_counter": getattr(agent, "play_action_counter", None),
            "action_counter": getattr(agent, "action_counter", None),
        }


async def _heartbeat_monitor(ws_server: BreakpointWebSocketServer) -> None:
    """Background task to monitor agent heartbeats and mark stale agents as disconnected."""
    while True:
        try:
            await asyncio.sleep(2.0)  # Check every 2 seconds
            current_time = time.time()
            state = ws_server._state
            stale_agents = []
            for agent_id, agent_state in list(state.agents.items()):
                if agent_state.status != "DISCONNECTED":
                    # If last_heartbeat is 0, agent just connected but hasn't sent first heartbeat yet
                    # Give it a grace period
                    if agent_state.last_heartbeat == 0.0:
                        # If agent connected more than timeout ago and still no heartbeat, mark stale
                        # This shouldn't happen normally, but handle it
                        continue
                    time_since_heartbeat = current_time - agent_state.last_heartbeat
                    if time_since_heartbeat > state._heartbeat_timeout:
                        stale_agents.append(agent_id)

            for agent_id in stale_agents:
                agent_state = state.agents.get(agent_id)
                if agent_state and agent_state.status != "DISCONNECTED":
                    logger.warning(
                        "Agent %s heartbeat timeout (%.1fs since last heartbeat), marking as DISCONNECTED",
                        agent_id,
                        current_time - agent_state.last_heartbeat,
                    )
                    await ws_server._handle_agent_disconnected(agent_id)
        except Exception as exc:
            logger.error("Error in heartbeat monitor: %s", exc, exc_info=True)


async def run_breakpoint_server(
    http_port: int = 8080,
    ws_port: int = 8765,
    static_dir: Optional[str] = None,
) -> None:
    """Run HTTP + websocket breakpoint server."""
    if static_dir is None:
        # __file__ is at arc-agi-3-benchmarking/src/arcagi3/breakpoint_server.py
        # Go up 3 levels to get to arc-agi-3-benchmarking/
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # Prefer breakpointer/frontend/dist (current UI), fall back to older names, then repo root.
        candidate_breakpointer_frontend = os.path.join(root, "breakpointer", "frontend", "dist")
        candidate_breakpointer = os.path.join(root, "breakpointer", "dist")
        candidate_breakpoint_ui = os.path.join(root, "breakpoint_ui", "dist")
        candidate_debug = os.path.join(root, "debug_ui", "dist")
        if os.path.isdir(candidate_breakpointer_frontend):
            static_dir = candidate_breakpointer_frontend
        elif os.path.isdir(candidate_breakpointer):
            static_dir = candidate_breakpointer
        elif os.path.isdir(candidate_breakpoint_ui):
            static_dir = candidate_breakpoint_ui
        elif os.path.isdir(candidate_debug):
            static_dir = candidate_debug
        else:
            static_dir = root

    state = BreakpointServerState()
    httpd = start_http_server(http_port, static_dir)

    ws_server = BreakpointWebSocketServer("0.0.0.0", ws_port, state)
    await ws_server.start()

    # Start heartbeat monitor
    monitor_task = asyncio.create_task(_heartbeat_monitor(ws_server))

    stop: asyncio.Future = asyncio.get_running_loop().create_future()

    def _handle_signal(*_: Any) -> None:
        if not stop.done():
            stop.set_result(None)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass

    await stop

    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    httpd.shutdown()
    logger.info("Breakpoint server stopped")


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="ARC-AGI-3 Breakpoint UI server (HTTP + WebSocket)"
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port for HTTP static server (default: 8080)",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="Port for WebSocket server (default: 8765)",
    )
    parser.add_argument(
        "--static-dir",
        type=str,
        default=None,
        help="Directory to serve static files from (default: breakpointer/dist if present)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_breakpoint_server(
            http_port=args.http_port,
            ws_port=args.ws_port,
            static_dir=args.static_dir,
        )
    )


if __name__ == "__main__":
    main()


