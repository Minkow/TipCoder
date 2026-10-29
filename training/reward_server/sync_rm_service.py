#!/usr/bin/env python3
"""Synchronous remote reward service that wraps ``LocalRMActor`` via HTTP.

This server keeps a single ``LocalRMActor`` instance warm and processes incoming
batches synchronously. Each HTTP request carries a mega-batch (the
ExperienceMaker already aggregated all prompts), while the actor internally
chunks work so the familiar ``RM chunk`` logs still appear.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional

from local_code_rm import LocalRMActor
from openrlhf.utils.rm_serialization import encode_for_transport

LOGGER = logging.getLogger("sync_rm_service")
RM_INSTANCE: Optional[LocalRMActor] = None
RM_INIT_LOCK = threading.Lock()
SERVICE_BOOT_TIME = time.time()


def _init_rm_actor() -> LocalRMActor:
    """Initialize the global LocalRMActor exactly once."""

    global RM_INSTANCE
    if RM_INSTANCE is not None:
        return RM_INSTANCE

    with RM_INIT_LOCK:
        if RM_INSTANCE is None:
            LOGGER.info("Initializing LocalRMActor (this may take a while)...")
            RM_INSTANCE = LocalRMActor()
            LOGGER.info("LocalRMActor initialization completed")
    return RM_INSTANCE


def _validate_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate request payload and support legacy field names."""

    if "query" in data and "queries" not in data:
        data = dict(data)
        data["queries"] = data.pop("query")

    required = ["queries", "prompts", "labels"]
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")

    queries = data["queries"]
    prompts = data["prompts"]
    labels = data["labels"]

    if (
        not isinstance(queries, list)
        or not isinstance(prompts, list)
        or not isinstance(labels, list)
    ):
        raise ValueError("queries, prompts, labels must be lists")
    if not queries:
        raise ValueError("queries cannot be empty")
    if not (len(queries) == len(prompts) == len(labels)):
        raise ValueError("queries, prompts, labels must have identical lengths")

    return {
        "queries": queries,
        "prompts": prompts,
        "labels": labels,
    }


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(
    handler: BaseHTTPRequestHandler, status: HTTPStatus, message: str
) -> None:
    LOGGER.warning("Request from %s failed: %s", handler.client_address, message)
    _send_json(handler, status.value, {"error": message})


class SyncRMHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server with bounded concurrency to protect the RM."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        max_concurrency: int,
        request_timeout: float,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.max_concurrency = max(1, max_concurrency)
        self.request_timeout = max(1.0, float(request_timeout))
        self.concurrency_limiter = threading.BoundedSemaphore(
            value=self.max_concurrency
        )
        self.total_requests = 0
        self.failed_requests = 0


class RMRequestHandler(BaseHTTPRequestHandler):
    server_version = "SyncRM/1.0"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        LOGGER.debug("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/healthz"):
            uptime = int(time.time() - SERVICE_BOOT_TIME)
            pending = self._pending_requests()
            _send_json(
                self,
                HTTPStatus.OK.value,
                {
                    "status": "ok",
                    "uptime_seconds": uptime,
                    "pending_requests": pending,
                    "total_requests": getattr(self.server, "total_requests", 0),
                    "failed_requests": getattr(self.server, "failed_requests", 0),
                },
            )
            return
        _send_error(self, HTTPStatus.NOT_FOUND, "endpoint not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path in ("/", "/score"):
            self._handle_score_request()
            return
        _send_error(self, HTTPStatus.NOT_FOUND, "endpoint not found")

    def _pending_requests(self) -> int:
        limiter = getattr(self.server, "concurrency_limiter", None)
        if limiter is None:
            return 0
        # BoundedSemaphore doesn't expose the current value, so approximate using the private counter.
        current_value = limiter._value if hasattr(limiter, "_value") else 0  # type: ignore[attr-defined]
        return max(0, self.server.max_concurrency - current_value)

    def _handle_score_request(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            _send_error(self, HTTPStatus.BAD_REQUEST, "missing request body")
            return

        try:
            raw_body = self.rfile.read(content_length)
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            _send_error(self, HTTPStatus.BAD_REQUEST, f"invalid json: {exc}")
            return

        try:
            payload = _validate_payload(data)
        except ValueError as exc:
            _send_error(self, HTTPStatus.BAD_REQUEST, str(exc))
            return

        limiter = getattr(self.server, "concurrency_limiter", None)
        acquired = False
        if limiter is not None:
            try:
                acquired = limiter.acquire(timeout=self.server.request_timeout)
            except RuntimeError:
                acquired = False
            if not acquired:
                _send_error(
                    self, HTTPStatus.SERVICE_UNAVAILABLE, "server is busy, please retry"
                )
                return

        start = time.perf_counter()
        self.server.total_requests += 1  # type: ignore[attr-defined]

        try:
            actor = _init_rm_actor()
            LOGGER.info("Processing batch of %d queries", len(payload["queries"]))
            raw_result = actor.score(
                payload["queries"], payload["prompts"], payload["labels"]
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            encoded_result = encode_for_transport(raw_result)
            if isinstance(encoded_result, dict):
                meta = encoded_result.setdefault("meta", {})
                meta.update(
                    {
                        "batch_size": len(payload["queries"]),
                        "duration_ms": round(duration_ms, 3),
                        "uptime_seconds": int(time.time() - SERVICE_BOOT_TIME),
                    }
                )
            _send_json(self, HTTPStatus.OK.value, encoded_result)
            LOGGER.info("Batch completed in %.2f ms", duration_ms)
        except Exception as exc:  # pylint: disable=broad-except
            self.server.failed_requests += 1  # type: ignore[attr-defined]
            LOGGER.error("Failed to process request: %s", exc, exc_info=True)
            _send_error(self, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        finally:
            if acquired and limiter is not None:
                limiter.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronous Remote RM HTTP service")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Maximum number of simultaneous scoring requests",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for an available worker slot before rejecting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(asctime)s] [%(levelname)s] %(message)s")

    LOGGER.info(
        "Starting synchronous Remote RM service on %s:%d (max-concurrency=%d)",
        args.host,
        args.port,
        args.max_concurrency,
    )

    try:
        _init_rm_actor()
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Failed to initialize LocalRMActor: %s", exc, exc_info=True)
        raise SystemExit(1) from exc

    server = SyncRMHTTPServer(
        (args.host, args.port),
        RMRequestHandler,
        max_concurrency=args.max_concurrency,
        request_timeout=args.request_timeout,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Received shutdown signal, stopping server...")
    finally:
        server.server_close()
        LOGGER.info("Remote RM service stopped")


if __name__ == "__main__":
    main()
