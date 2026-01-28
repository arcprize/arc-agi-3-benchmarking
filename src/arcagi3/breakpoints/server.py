from __future__ import annotations

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
from typing import Any, Dict, List, Optional

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class StaticFileHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: Optional[str] = None, **kwargs):
        if directory is None:
            directory = getattr(self.__class__, "directory", os.getcwd())
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def start_http_server(port: int, static_dir: str) -> ThreadingTCPServer:
    # Create a handler class with the directory bound
    class Handler(StaticFileHandler):
        directory = static_dir

    httpd = ThreadingTCPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(
        target=httpd.serve_forever,
        name=f"BreakpointHTTPServer:{port}",
        daemon=True,
    )
    thread.start()
    logger.info("HTTP server serving %s on http://localhost:%d", static_dir, port)
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
    breakpoints: Dict[str, bool] = field(default_factory=dict)
    last_heartbeat: float = 0.0
    ws: Optional[WebSocketServerProtocol] = None
    play_num: Optional[int] = None
    play_action_counter: Optional[int] = None
    action_counter: Optional[int] = None
    schema: Dict[str, Any] = field(default_factory=dict)


class BreakpointServerState:
    def __init__(self) -> None:
        self.global_paused: bool = False
        self.global_breakpoints: Dict[str, bool] = {}
        self.agents: Dict[str, AgentState] = {}
        self.ui_clients: List[WebSocketServerProtocol] = []
        self.pending: Dict[str, asyncio.Future] = {}
        self.pending_continues: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._heartbeat_timeout: float = 10.0

    @staticmethod
    def _req_step_key(agent_id: str, request_id: str) -> str:
        return f"{agent_id}:{request_id}"


class BreakpointWebSocketServer:
    def __init__(self, host: str, port: int, state: BreakpointServerState) -> None:
        self._host = host
        self._port = port
        self._state = state
        self._server: Optional[websockets.server.Serve] = None
        self._pending_tasks: Dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
        )
        logger.info("Breakpoint WS server listening on ws://%s:%d", self._host, self._port)

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
                    continue
                if client_kind == "ui":
                    await self._handle_ui_message(ws, data)
                elif client_kind == "agent":
                    await self._handle_agent_message(ws, data)
        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError):
            pass
        except Exception as exc:
            logger.error("WebSocket handler error: %s", exc, exc_info=True)
        finally:
            if client_kind == "ui":
                await self._unregister_ui(ws)
            elif client_kind == "agent" and client_agent_id:
                state = self._state.agents.get(client_agent_id)
                if state and state.ws == ws:
                    await self._handle_agent_disconnected(client_agent_id)

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
                "paused": self._state.global_paused,
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

    async def _handle_ui_message(self, ws: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        msg_type = data.get("type")
        if msg_type == "set_global_state":
            self._state.global_paused = bool(data.get("paused", self._state.global_paused))
            breakpoints = data.get("breakpoints", self._state.global_breakpoints)
            self._state.global_breakpoints.update({k: bool(v) for k, v in breakpoints.items()})
            await self._broadcast(
                {
                    "type": "global_state_updated",
                    "global": {
                        "paused": self._state.global_paused,
                        "breakpoints": self._state.global_breakpoints,
                    },
                }
            )
        elif msg_type == "set_agent_breakpoints":
            agent_id = data.get("agent_id")
            bp = data.get("breakpoints", {})
            if agent_id and agent_id in self._state.agents:
                self._state.agents[agent_id].breakpoints.update({k: bool(v) for k, v in bp.items()})
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
                task_keys_to_remove = [
                    key
                    for key in list(self._pending_tasks.keys())
                    if key.startswith(f"{agent_id}:")
                ]
                for key in task_keys_to_remove:
                    task = self._pending_tasks.pop(key, None)
                    if task and not task.done():
                        task.cancel()
                state = self._state.agents.get(agent_id)
                if state and state.current_breakpoint:
                    req_id = state.current_breakpoint.get("request_id")
                    if req_id:
                        fut = self._state.pending.pop(req_id, None)
                        if fut and not fut.done():
                            fut.cancel()
                self._state.pending_continues.pop(agent_id, None)
                del self._state.agents[agent_id]
                await self._broadcast({"type": "agent_removed", "agent_id": agent_id})

    async def _handle_agent_hello(self, ws: WebSocketServerProtocol, msg: Dict[str, Any]) -> None:
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
        state.schema = msg.get("schema", state.schema) or {}
        incoming_breakpoints = msg.get("breakpoints", {}) or {}
        if incoming_breakpoints:
            state.breakpoints.update({k: bool(v) for k, v in incoming_breakpoints.items()})

        for point_id in _collect_point_ids(state.schema):
            state.breakpoints.setdefault(point_id, True)
            self._state.global_breakpoints.setdefault(point_id, False)

        state.status = "CONNECTED"
        state.ws = ws
        state.last_heartbeat = time.time()

        await self._broadcast({"type": "agent_updated", "agent": self._agent_to_dict(state)})
        await ws.send(
            json.dumps(
                {
                    "type": "hello_ack",
                    "global": {
                        "paused": self._state.global_paused,
                        "breakpoints": self._state.global_breakpoints,
                    },
                }
            )
        )

        queued = self._state.pending_continues.get(agent_id) or {}
        if queued:
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
            state.ws = ws
            state.last_heartbeat = time.time()
            if "config" in data and data.get("config") is not None:
                state.config = data.get("config")
            if "card_id" in data and data.get("card_id") is not None:
                state.card_id = data.get("card_id")
            if "game_id" in data and data.get("game_id") is not None:
                state.game_id = data.get("game_id")
            if "schema" in data and data.get("schema") is not None:
                state.schema = data.get("schema")
            state.status = data.get("status", state.status)
            state.score = int(data.get("score", state.score))
            state.last_step = data.get("step_name", state.last_step)
            if "play_num" in data and data.get("play_num") is not None:
                state.play_num = data.get("play_num")
            if "play_action_counter" in data and data.get("play_action_counter") is not None:
                state.play_action_counter = data.get("play_action_counter")
            if "action_counter" in data and data.get("action_counter") is not None:
                state.action_counter = data.get("action_counter")
            await self._broadcast({"type": "agent_updated", "agent": self._agent_to_dict(state)})

        elif msg_type == "breakpoint_request":
            point_id = data.get("point_id")
            payload = data.get("payload") or {}
            request_id = data.get("request_id")
            if not point_id or not request_id:
                return

            state = self._state.agents.get(agent_id)
            if not state:
                state = AgentState(agent_id=agent_id)
                self._state.agents[agent_id] = state

            state.ws = ws
            state.last_heartbeat = time.time()

            should_pause = (
                self._state.global_paused
                or bool(self._state.global_breakpoints.get(point_id, False))
                or bool(state.breakpoints.get(point_id, False))
            )

            if not should_pause:
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "breakpoint_continue",
                                "request_id": request_id,
                                "point_id": point_id,
                                "payload": payload,
                            }
                        )
                    )
                except Exception:
                    logger.info(
                        "Failed to auto-continue breakpoint for agent %s (point_id=%s)",
                        agent_id,
                        point_id,
                        exc_info=True,
                    )
                return

            state.status = "PAUSED"
            state.current_breakpoint = {
                "request_id": request_id,
                "point_id": point_id,
                "payload": payload,
            }
            await self._broadcast(
                {
                    "type": "breakpoint_pending",
                    "agent": self._agent_to_dict(state),
                    "point_id": point_id,
                    "request_id": request_id,
                    "payload": payload,
                }
            )

            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._state.pending[request_id] = fut
            task_key = self._state._req_step_key(agent_id, request_id)
            old_task = self._pending_tasks.pop(task_key, None)
            if old_task and not old_task.done():
                old_task.cancel()
            self._pending_tasks[task_key] = asyncio.create_task(
                self._breakpoint_wait_and_continue(
                    agent_id=agent_id,
                    point_id=point_id,
                    request_id=request_id,
                    original_payload=payload,
                    fut=fut,
                )
            )

        elif msg_type == "heartbeat":
            state = self._state.agents.get(agent_id)
            if state:
                state.ws = ws
                state.last_heartbeat = time.time()
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
        task_keys_to_remove = [
            key for key in list(self._pending_tasks.keys()) if key.startswith(f"{agent_id}:")
        ]
        for key in task_keys_to_remove:
            task = self._pending_tasks.pop(key, None)
            if task and not task.done():
                task.cancel()
        if old_status != "DISCONNECTED":
            await self._broadcast({"type": "agent_updated", "agent": self._agent_to_dict(state)})

    async def _breakpoint_wait_and_continue(
        self,
        agent_id: str,
        point_id: str,
        request_id: str,
        original_payload: Dict[str, Any],
        fut: asyncio.Future,
    ) -> None:
        task_key = self._state._req_step_key(agent_id, request_id)
        try:
            overrides = await fut
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
                    "point_id": point_id,
                    "request_id": request_id,
                }
            )

            payload_to_send = overrides if overrides is not None else original_payload
            ws_to_use: Optional[WebSocketServerProtocol] = state.ws if state else None
            if ws_to_use is None:
                self._state.pending_continues.setdefault(agent_id, {})[request_id] = payload_to_send
                return
            try:
                await ws_to_use.send(
                    json.dumps(
                        {
                            "type": "breakpoint_continue",
                            "request_id": request_id,
                            "point_id": point_id,
                            "payload": payload_to_send,
                        }
                    )
                )
            except Exception:
                self._state.pending_continues.setdefault(agent_id, {})[request_id] = payload_to_send
        except asyncio.CancelledError:
            pass
        finally:
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
            "schema": agent.schema,
        }


def _collect_point_ids(schema: Dict[str, Any]) -> List[str]:
    point_ids: List[str] = []
    if not schema:
        return point_ids
    for section in schema.get("sections", []):
        for point in section.get("points", []):
            pid = point.get("point_id")
            if pid:
                point_ids.append(pid)
    return point_ids


async def _heartbeat_monitor(ws_server: BreakpointWebSocketServer) -> None:
    while True:
        await asyncio.sleep(2.0)
        current_time = time.time()
        state = ws_server._state
        stale_agents = []
        for agent_id, agent_state in list(state.agents.items()):
            if agent_state.status != "DISCONNECTED" and agent_state.last_heartbeat:
                if current_time - agent_state.last_heartbeat > state._heartbeat_timeout:
                    stale_agents.append(agent_id)
        for agent_id in stale_agents:
            await ws_server._handle_agent_disconnected(agent_id)


async def run_breakpoint_server(
    http_port: int = 8080,
    ws_port: int = 8765,
    static_dir: Optional[str] = None,
) -> None:
    if static_dir is None:
        # Go up from src/arcagi3/breakpoints/server.py to project root
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        candidate_ui = os.path.join(root, "breakpointer", "dist")
        if os.path.isdir(candidate_ui):
            static_dir = candidate_ui
            logger.info("Found UI build directory: %s", static_dir)
        else:
            static_dir = root
            logger.warning(
                "UI build directory not found at %s, serving from project root: %s",
                candidate_ui,
                static_dir,
            )

    httpd = start_http_server(http_port, static_dir)
    state = BreakpointServerState()
    ws_server = BreakpointWebSocketServer("0.0.0.0", ws_port, state)
    await ws_server.start()
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
