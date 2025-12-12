import asyncio
import json
import logging
import threading
import uuid
from typing import Any, Dict, Optional, Literal

import websockets
from PIL import Image

from .agent import MultimodalAgent, FrameImageSequence, FrameGridSequence, FrameGrid
from .utils.image import image_to_base64, base64_to_image, image_diff, grid_to_image


logger = logging.getLogger(__name__)


class BreakpointMultimodalAgent(MultimodalAgent):
    """
    Multimodal agent that can pause at configured breakpoints and communicate
    with the local breakpoint websocket server.

    This wraps the core substeps of `MultimodalAgent` plus `_execute_game_action`
    and sends their inputs to the breakpoint server before executing, allowing a UI
    to inspect/modify them.
    """

    def __init__(
        self,
        *args: Any,
        breakpoint_ws_url: str = "ws://localhost:8765/ws",
        agent_kind: str = "multimodal",
        enable_breakpoints: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.breakpoint_ws_url = breakpoint_ws_url
        self.agent_kind = agent_kind
        self.enable_breakpoints = enable_breakpoints
        self.agent_id = uuid.uuid4().hex

        # Internal asyncio loop for websocket communication
        self._ws_loop = asyncio.new_event_loop()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_lock = threading.Lock()
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, name=f"BreakpointWS:{self.agent_id}", daemon=True
        )
        self._ws_thread.start()

        # Pending breakpoint futures: (request_id) -> Future[payload]
        self._pending_breakpoints: Dict[str, asyncio.Future] = {}

        # Wait until connection is established before allowing steps to execute
        # When breakpointer is enabled, agent MUST wait for server before proceeding
        self._connected_event = threading.Event()
        self._schedule_coroutine(self._connect_and_handshake())
        logger.info(
            "[%s] Waiting for breakpoint server at %s... "
            "Start the server with: python scripts/run_breakpoint_server.py",
            self.agent_id,
            self.breakpoint_ws_url,
        )
        self._connected_event.wait()
        logger.info("[%s] Connected to breakpoint server, proceeding with game", self.agent_id)
        
        # Track previous frame for breakpointer display
        self._breakpointer_previous_frame_image: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Websocket connection helpers
    # ------------------------------------------------------------------

    def _run_ws_loop(self) -> None:
        asyncio.set_event_loop(self._ws_loop)
        self._ws_loop.run_forever()

    def _schedule_coroutine(self, coro: Any) -> None:
        asyncio.run_coroutine_threadsafe(coro, self._ws_loop)

    async def _send_agent_update(
        self,
        status: Optional[str] = None,
        step_name: Optional[str] = None,
        score: Optional[int] = None,
    ) -> None:
        """Best-effort agent_update to keep server/UI in sync (especially game_id)."""
        with self._ws_lock:
            ws = self._ws
        if not ws:
            return
        payload: Dict[str, Any] = {
            "type": "agent_update",
            "agent_id": self.agent_id,
            "config": self.config,
            "card_id": self.card_id,
            "game_id": self.current_game_id,
            "play_num": getattr(self, "_current_play", 1),
            "play_action_counter": getattr(self, "_play_action_counter", 0),
            "action_counter": getattr(self, "action_counter", 0),
        }
        if status is not None:
            payload["status"] = status
        if step_name is not None:
            payload["step_name"] = step_name
        if score is not None:
            payload["score"] = score
        try:
            await ws.send(json.dumps(payload))
        except Exception:
            # Don't crash agent if UI/server isn't available
            logger.debug("[%s] Failed to send agent_update", self.agent_id, exc_info=True)

    async def _connect_and_handshake(self) -> None:
        """Connect to breakpoint server, retrying until available."""
        attempt = 0
        while True:
            try:
                attempt += 1
                if attempt == 1:
                    logger.info(
                        "[%s] Attempting to connect to breakpoint server at %s...",
                        self.agent_id,
                        self.breakpoint_ws_url,
                    )
                else:
                    logger.info(
                        "[%s] Retrying connection to breakpoint server (attempt %d)...",
                        self.agent_id,
                        attempt,
                    )
                # Configure websocket client with ping/pong keepalive to prevent premature disconnections
                ws = await websockets.connect(
                    self.breakpoint_ws_url,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong response
                    close_timeout=10,  # Wait 10 seconds for close handshake
                )
                with self._ws_lock:
                    self._ws = ws

                hello = {
                    "client": "agent",
                    "type": "agent_hello",
                    "agent_id": self.agent_id,
                    "config": self.config,
                    "card_id": self.card_id,
                    "game_id": self.current_game_id,
                }
                await ws.send(json.dumps(hello))

                # Best-effort ack
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    _ = json.loads(raw)
                except Exception:
                    pass

                self._connected_event.set()
                logger.info(
                    "[%s] Successfully connected to breakpoint server", self.agent_id
                )

                # Start heartbeat task
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
                    "[%s] Breakpoint server not available: %s. "
                    "Waiting 2 seconds before retry...",
                    self.agent_id,
                    exc,
                )
                await asyncio.sleep(2.0)

    async def _heartbeat_loop(
        self, ws: websockets.WebSocketClientProtocol
    ) -> None:
        """Send periodic heartbeat messages to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(5.0)  # Send heartbeat every 5 seconds
                try:
                    msg = {
                        "type": "heartbeat",
                        "agent_id": self.agent_id,
                    }
                    await ws.send(json.dumps(msg))
                except Exception:
                    # Connection lost, heartbeat loop will exit
                    break
        except asyncio.CancelledError:
            pass

    async def _ws_reader_loop(
        self, ws: websockets.WebSocketClientProtocol
    ) -> None:
        """Read all messages from websocket and route them to waiting futures."""
        try:
            async for raw_msg in ws:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                if msg_type == "breakpoint_continue":
                    request_id = data.get("request_id")
                    payload = data.get("payload")
                    if not request_id:
                        continue
                    fut = self._pending_breakpoints.pop(request_id, None)
                    if fut and not fut.done():
                        fut.set_result(payload)
        except Exception as exc:
            logger.info(
                "[%s] Websocket connection closed: %s, reconnecting...",
                self.agent_id,
                exc,
            )
        finally:
            # Cancel any pending breakpoints
            for fut in self._pending_breakpoints.values():
                if not fut.done():
                    fut.cancel()
            self._pending_breakpoints.clear()

            with self._ws_lock:
                self._ws = None
            self._connected_event.clear()
            self._schedule_coroutine(self._connect_and_handshake())

    async def _await_breakpoint_async(
        self, step: str, phase: Literal["pre", "post"], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send breakpoint request and wait for continue message with overrides."""
        if not self.enable_breakpoints:
            return payload

        # Wait for a connection if needed
        while not self._connected_event.is_set():
            await asyncio.sleep(0.5)

        with self._ws_lock:
            ws = self._ws

        if not ws:
            return payload

        # Create a future for this breakpoint request
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        request_id = uuid.uuid4().hex
        self._pending_breakpoints[request_id] = fut

        try:
            # Keep server updated with game_id before we pause
            await self._send_agent_update(status="PAUSED", step_name=f"{step}_{phase}")
            msg = {
                "type": "breakpoint_request",
                "agent_id": self.agent_id,
                "request_id": request_id,
                "step": step,
                "phase": phase,
                "point": f"{step}_{phase}",
                "payload": payload,
            }
            await ws.send(json.dumps(msg))

            # Wait for the future to be resolved by the reader loop
            result = await fut
            # After continue, we're running again
            await self._send_agent_update(status="RUNNING", step_name=f"{step}_{phase}")
            return result if result is not None else payload
        except asyncio.CancelledError:
            # Connection was lost, return original payload
            return payload
        finally:
            # Clean up the future if it's still there
            self._pending_breakpoints.pop(request_id, None)

    def _await_breakpoint(
        self, step: str, phase: Literal["pre", "post"], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            self._await_breakpoint_async(step, phase, payload), self._ws_loop
        )
        return future.result()

    def play_game(self, game_id: str, resume_from_checkpoint: bool = False):
        """
        Override to ensure breakpoint server sees the game_id immediately.
        """
        # Set as early as possible
        self.current_game_id = game_id
        # Best-effort notify server
        self._schedule_coroutine(self._send_agent_update(status="RUNNING", step_name="play_game"))
        return super().play_game(game_id=game_id, resume_from_checkpoint=resume_from_checkpoint)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_images(images: FrameImageSequence) -> Any:
        encoded = []
        for img in images:
            try:
                buf = image_to_base64(img)
                encoded.append(
                    {
                        "kind": "image",
                        "width": img.width,
                        "height": img.height,
                        "data": buf,
                    }
                )
            except Exception:
                encoded.append({"kind": "image", "error": "encode_failed"})
        return encoded

    @classmethod
    def _encode_latest_frame_image(cls, images: Optional[FrameImageSequence]) -> Optional[Dict[str, Any]]:
        """Encode the latest frame image (last image in the sequence)."""
        if not images:
            return None
        try:
            return cls._encode_images([images[-1]])[0]
        except Exception:
            return None

    @staticmethod
    def _decode_image(obj: Any) -> Optional[Image.Image]:
        """Decode a single image payload object {kind:'image', data:'...'} into a PIL image."""
        if not isinstance(obj, dict):
            return None
        if obj.get("kind") != "image":
            return None
        data = obj.get("data")
        if not isinstance(data, str) or not data:
            return None
        try:
            return base64_to_image(data)
        except Exception:
            return None

    @classmethod
    def _decode_image_list(cls, arr: Any) -> Optional[FrameImageSequence]:
        """Decode an array of image payload objects into a list of PIL images."""
        if not isinstance(arr, list):
            return None
        out: list[Image.Image] = []
        for item in arr:
            img = cls._decode_image(item)
            if img is None:
                return None
            out.append(img)
        return out

    # ------------------------------------------------------------------
    # Overridden substeps with breakpoint hooks
    # ------------------------------------------------------------------

    def analyze_outcome_step(
        self,
        current_frame_images: FrameImageSequence,
        current_frame_grids: FrameGridSequence,
        current_score: int,
    ) -> str:
        # Get previous frame - use _previous_images if available, otherwise use tracked previous frame
        # If vision is disabled, convert grids to images for breakpointer display
        if self._use_vision and self._previous_images:
            previous_frame_img = self._previous_images[-1]
        elif current_frame_grids and len(self._previous_grids) > 0:
            # Vision disabled - convert previous grid to image for breakpointer
            previous_frame_img = grid_to_image(self._previous_grids[-1])
        else:
            previous_frame_img = None
            
        # Encode frames for breakpointer display
        if self._use_vision:
            current_frame = self._encode_latest_frame_image(current_frame_images)
        else:
            # Vision disabled - convert current grid to image for breakpointer
            if current_frame_grids:
                current_img = grid_to_image(current_frame_grids[-1])
                current_frame = self._encode_images([current_img])[0] if current_img else None
            else:
                current_frame = None
                
        previous_frame = self._breakpointer_previous_frame_image
        if previous_frame_img:
            previous_frame = self._encode_images([previous_frame_img])[0]
        
        # Create helper image if enabled and we have both previous and current frames
        helper_image = None
        if self._include_helper_image and previous_frame and current_frame:
            try:
                # Decode previous and current frames to create diff
                prev_img = self._decode_image(previous_frame)
                curr_img = self._decode_image(current_frame)
                if prev_img and curr_img:
                    diff_img = image_diff(prev_img, curr_img)
                    helper_image = {
                        "kind": "image",
                        "width": diff_img.width,
                        "height": diff_img.height,
                        "data": image_to_base64(diff_img),
                    }
            except Exception as e:
                logger.debug(f"Failed to create helper image: {e}")
        
        # Store latest frame for steps that don't have direct access to frame images (e.g. execute_action)
        self._breakpointer_latest_frame_image = current_frame
        # Store current frame as previous for next iteration
        self._breakpointer_previous_frame_image = current_frame
        
        analysis = super().analyze_outcome_step(
            current_frame_images=current_frame_images,
            current_frame_grids=current_frame_grids,
            current_score=current_score,
        )
        post_payload: Dict[str, Any] = {
            "analysis": analysis,
            "previous_frame_image": previous_frame,
            "helper_image": helper_image,
            "current_frame_image": current_frame,
            "latest_frame_image": current_frame,  # Keep for backward compatibility
            "memory_text": self._memory_prompt,
            "memory_word_limit": self.memory_word_limit,
        }
        post_updated = self._await_breakpoint("analyze", "post", post_payload)
        if "analysis" in post_updated and isinstance(post_updated.get("analysis"), str):
            analysis = post_updated["analysis"]
        if "memory_text" in post_updated and isinstance(post_updated.get("memory_text"), str):
            self._memory_prompt = post_updated["memory_text"]
        if "memory_word_limit" in post_updated:
            try:
                self.memory_word_limit = int(post_updated["memory_word_limit"])
            except Exception:
                pass
        return analysis

    def decide_human_action_step(
        self,
        frame_images: FrameImageSequence,
        frame_grids: FrameGridSequence,
        analysis: str,
    ) -> Dict[str, Any]:
        latest = self._encode_latest_frame_image(frame_images)
        self._breakpointer_latest_frame_image = latest
        result = super().decide_human_action_step(
            frame_images=frame_images,
            frame_grids=frame_grids,
            analysis=analysis,
        )
        post_payload: Dict[str, Any] = {
            "result": result,
            "latest_frame_image": latest,
            "memory_text": self._memory_prompt,
            "memory_word_limit": self.memory_word_limit,
        }
        post_updated = self._await_breakpoint("decide", "post", post_payload)
        if isinstance(post_updated.get("result"), dict):
            result = post_updated["result"]
        if "memory_text" in post_updated and isinstance(post_updated.get("memory_text"), str):
            self._memory_prompt = post_updated["memory_text"]
        if "memory_word_limit" in post_updated:
            try:
                self.memory_word_limit = int(post_updated["memory_word_limit"])
            except Exception:
                pass
        return result

    def convert_human_to_game_action_step(
        self,
        human_action: str,
        last_frame_image: Image.Image,
        last_frame_grid: FrameGrid,
    ) -> Dict[str, Any]:
        latest = self._encode_images([last_frame_image])[0]
        self._breakpointer_latest_frame_image = latest
        result = super().convert_human_to_game_action_step(
            human_action=human_action,
            last_frame_image=last_frame_image,
            last_frame_grid=last_frame_grid,
        )
        post_payload: Dict[str, Any] = {
            "result": result,
            "latest_frame_image": latest,
            "memory_text": self._memory_prompt,
            "memory_word_limit": self.memory_word_limit,
        }
        post_updated = self._await_breakpoint("convert", "post", post_payload)
        if isinstance(post_updated.get("result"), dict):
            result = post_updated["result"]
        if "memory_text" in post_updated and isinstance(post_updated.get("memory_text"), str):
            self._memory_prompt = post_updated["memory_text"]
        if "memory_word_limit" in post_updated:
            try:
                self.memory_word_limit = int(post_updated["memory_word_limit"])
            except Exception:
                pass
        return result

    def _execute_game_action(
        self,
        action_name: str,
        action_data: Optional[Dict[str, Any]],
        game_id: str,
        guid: Optional[str],
        reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action": action_name,
            "action_data": action_data or {},
            "game_id": game_id,
            "guid": guid,
            "reasoning": reasoning,
            # Best effort: we don't have the current frames here, so we include the last-seen frame image.
            "latest_frame_image": getattr(self, "_breakpointer_latest_frame_image", None),
            "memory_text": self._memory_prompt,
            "memory_word_limit": self.memory_word_limit,
        }
        updated = self._await_breakpoint("execute_action", "pre", payload)
        action_name = updated.get("action", action_name)
        action_data = updated.get("action_data", action_data)
        game_id = updated.get("game_id", game_id)
        guid = updated.get("guid", guid)
        reasoning = updated.get("reasoning", reasoning)
        if "memory_text" in updated and isinstance(updated.get("memory_text"), str):
            self._memory_prompt = updated["memory_text"]
        if "memory_word_limit" in updated:
            try:
                self.memory_word_limit = int(updated["memory_word_limit"])
            except Exception:
                pass
        result = super()._execute_game_action(
            action_name=action_name,
            action_data=action_data,
            game_id=game_id,
            guid=guid,
            reasoning=reasoning,
        )
        post_payload: Dict[str, Any] = {
            "result": result,
            "latest_frame_image": getattr(self, "_breakpointer_latest_frame_image", None),
            "memory_text": self._memory_prompt,
            "memory_word_limit": self.memory_word_limit,
        }
        post_updated = self._await_breakpoint("execute_action", "post", post_payload)
        if isinstance(post_updated.get("result"), dict):
            result = post_updated["result"]
        if "memory_text" in post_updated and isinstance(post_updated.get("memory_text"), str):
            self._memory_prompt = post_updated["memory_text"]
        if "memory_word_limit" in post_updated:
            try:
                self.memory_word_limit = int(post_updated["memory_word_limit"])
            except Exception:
                pass
        return result


