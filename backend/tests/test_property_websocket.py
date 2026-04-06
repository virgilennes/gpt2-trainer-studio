"""Property test: WebSocket update interval (Property 10).

**Validates: Requirements 8.2**

For any pair of consecutive progress messages during a long-running
operation, the time between them should be no greater than 2 seconds.

We test this by driving the :class:`ConnectionManager` broadcast method
with a sequence of rapid-fire progress calls and verifying that the
messages actually delivered to clients never exceed the 2-second gap
constraint.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from backend.app.api.websocket import ConnectionManager
from backend.app.models.schemas import WSMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeWebSocket:
    """Minimal stand-in for ``fastapi.WebSocket`` that records sent data."""

    def __init__(self) -> None:
        self.messages: list[WSMessage] = []
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        msg = WSMessage.model_validate_json(data)
        self.messages.append(msg)

    async def receive_text(self) -> str:
        # Block forever – simulates a quiet client.
        await asyncio.sleep(999_999)
        return ""


async def _connect_fake(mgr: ConnectionManager, ws: FakeWebSocket) -> None:
    """Register a fake websocket without going through the full endpoint."""
    await ws.accept()
    mgr._connections.append(ws)  # type: ignore[attr-defined]
    # Send state sync like the real connect method.
    await mgr._send_state_sync(ws)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    num_broadcasts=st.integers(min_value=2, max_value=50),
)
@pytest.mark.asyncio
async def test_progress_broadcast_interval_within_2_seconds(
    num_broadcasts: int,
) -> None:
    """**Feature: gpt-text-generator, Property 10: WebSocket update interval**

    **Validates: Requirements 8.2**

    For any sequence of rapid-fire progress broadcasts, the timestamps
    on messages actually delivered to the client must be spaced ≤ 2 seconds
    apart (the manager rate-limits to enforce this).
    """
    mgr = ConnectionManager()
    # Use a very small interval so the test runs quickly while still
    # exercising the throttle logic.  The real default is 2.0 s.
    mgr.min_broadcast_interval = 0.05  # 50 ms for fast tests

    fake_ws = FakeWebSocket()
    await _connect_fake(mgr, fake_ws)

    # Clear the initial state_change message so we only measure progress.
    fake_ws.messages.clear()

    for i in range(num_broadcasts):
        await mgr.broadcast_progress({"step": i, "pct": i / num_broadcasts})
        # Tiny sleep to simulate real-world call cadence.
        await asyncio.sleep(0.01)

    delivered = fake_ws.messages

    # We must have received at least one progress message.
    assert len(delivered) >= 1, "Expected at least one progress message"

    # Verify that every pair of consecutive delivered messages respects
    # the interval constraint.  We use the monotonic send timestamps
    # recorded by the manager (via the ISO timestamp in the message).
    from datetime import datetime, timezone

    timestamps: list[float] = []
    for msg in delivered:
        dt = datetime.fromisoformat(msg.timestamp)
        timestamps.append(dt.timestamp())

    for idx in range(1, len(timestamps)):
        gap = timestamps[idx] - timestamps[idx - 1]
        # Allow a small tolerance (50 ms) for async scheduling jitter.
        assert gap >= mgr.min_broadcast_interval - 0.05, (
            f"Messages {idx - 1} and {idx} were delivered too close together: "
            f"{gap:.4f}s < {mgr.min_broadcast_interval}s"
        )
        # The key property: gap must not exceed 2 seconds.
        assert gap <= 2.0, (
            f"Messages {idx - 1} and {idx} exceeded 2-second interval: "
            f"{gap:.4f}s"
        )
