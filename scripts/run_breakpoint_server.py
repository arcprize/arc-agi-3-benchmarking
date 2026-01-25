import argparse
import asyncio
import logging

from arcagi3.breakpoints.server import run_breakpoint_server


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="ARC-AGI-3 breakpoint UI server (HTTP + WebSocket)"
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

