from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from typing import Any, Dict, Optional

import websockets

from arcagi3.breakpoints.spec import BreakpointSpec

logger = logging.getLogger(__name__)


class BreakpointClient:
    """
    WebSocket client for breakpoint communication with the UI server.
    """

    def __init__(
        self,
        *,
        breakpoint_ws_url: str = "ws://localhost:8765/ws",
        enable_breakpoints: bool = True,
        agent_id: Optional[str] = None,
    ) -> None:
        self.breakpoint_ws_url = breakpoint_ws_url
        self.enable_breakpoints = enable_breakpoints
        self.agent_id = agent_id or uuid.uuid4().hex

        self._schema: Dict[str, Any] = {}
        self._breakpoints: Dict[str, bool] = {}
        self._identity: Dict[str, Any] = {}

        self._ws_loop = asyncio.new_event_loop()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_lock = threading.Lock()
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, name=f"BreakpointWS:{self.agent_id}", daemon=True
        )
        self._ws_thread.start()

        self._pending_breakpoints: Dict[str, asyncio.Future] = {}
        self._connected_event = threading.Event()
        self._schedule_coroutine(self._connect_and_handshake())

    def set_schema(self, spec: Optional[BreakpointSpec]) -> None:
        if spec:
            self._schema = spec.to_dict()
            for point_id in spec.point_ids():
                self._breakpoints.setdefault(point_id, True)
        else:
            self._schema = {}

    def set_breakpoints(self, breakpoints: Dict[str, bool]) -> None:
        self._breakpoints.update({k: bool(v) for k, v in breakpoints.items()})

    def set_identity(
        self,
        *,
        config: Optional[str] = None,
        card_id: Optional[str] = None,
        game_id: Optional[str] = None,
    ) -> None:
        if config is not None:
            self._identity["config"] = config
        if card_id is not None:
            self._identity["card_id"] = card_id
        if game_id is not None:
            self._identity["game_id"] = game_id

    def _run_ws_loop(self) -> None:
        asyncio.set_event_loop(self._ws_loop)
        self._ws_loop.run_forever()

    def _schedule_coroutine(self, coro: Any) -> None:
        asyncio.run_coroutine_threadsafe(coro, self._ws_loop)

    async def _connect_and_handshake(self) -> None:
        attempt = 0
        while True:
            try:
                attempt += 1
                ws = await websockets.connect(
                    self.breakpoint_ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                )
                with self._ws_lock:
                    self._ws = ws

                hello = {
                    "client": "agent",
                    "type": "agent_hello",
                    "agent_id": self.agent_id,
                    "schema": self._schema,
                    "breakpoints": self._breakpoints,
                }
                hello.update(self._identity)
                await ws.send(json.dumps(hello))

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    _ = json.loads(raw)
                except Exception:
                    pass

                self._connected_event.set()
                heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
                try:
                    await self._ws_reader_loop(ws)
                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                return
            except Exception as exc:
                logger.warning(
                    "[%s] Breakpoint server not available: %s. Retrying...",
                    self.agent_id,
                    exc,
                )
                await asyncio.sleep(2.0)

    async def _heartbeat_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        try:
            while True:
                await asyncio.sleep(5.0)
                try:
                    msg = {"type": "heartbeat", "agent_id": self.agent_id}
                    await ws.send(json.dumps(msg))
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    async def _ws_reader_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        try:
            async for raw_msg in ws:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "breakpoint_continue":
                    request_id = data.get("request_id")
                    payload = data.get("payload")
                    fut = self._pending_breakpoints.pop(request_id, None)
                    if fut and not fut.done():
                        fut.set_result(payload)
        finally:
            for fut in self._pending_breakpoints.values():
                if not fut.done():
                    fut.cancel()
            self._pending_breakpoints.clear()
            with self._ws_lock:
                self._ws = None
            self._connected_event.clear()
            self._schedule_coroutine(self._connect_and_handshake())

    async def _send_agent_update(self, payload: Dict[str, Any]) -> None:
        with self._ws_lock:
            ws = self._ws
        if not ws:
            return
        try:
            await ws.send(json.dumps(payload))
        except Exception:
            logger.debug("[%s] Failed to send agent_update", self.agent_id, exc_info=True)

    def send_agent_update(self, payload: Dict[str, Any]) -> None:
        self._schedule_coroutine(self._send_agent_update(payload))

    async def await_breakpoint_async(
        self, point_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.enable_breakpoints:
            return payload
        while not self._connected_event.is_set():
            await asyncio.sleep(0.5)
        with self._ws_lock:
            ws = self._ws
        if not ws:
            return payload

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        request_id = uuid.uuid4().hex
        self._pending_breakpoints[request_id] = fut

        try:
            msg = {
                "type": "breakpoint_request",
                "agent_id": self.agent_id,
                "request_id": request_id,
                "point_id": point_id,
                "payload": payload,
            }
            await ws.send(json.dumps(msg))
            result = await fut
            return result if result is not None else payload
        except asyncio.CancelledError:
            return payload
        finally:
            self._pending_breakpoints.pop(request_id, None)

    def await_breakpoint(self, point_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            self.await_breakpoint_async(point_id, payload), self._ws_loop
        )
        return future.result()
