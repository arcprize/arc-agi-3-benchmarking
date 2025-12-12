"""
CLI runner for the ARC breakpoint WebSocket + HTTP server and React UI.

Usage:
    python run_breakpoint_server.py --http-port 8080 --ws-port 8765
"""
import sys
import os

# Add src to path (go up one level from scripts/ to get to arc-agi-3-benchmarking/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from arcagi3.breakpoint_server import main

if __name__ == "__main__":
    main()

