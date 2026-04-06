"""Property-based tests for the Demo Orchestrator.

Tests Properties 7, 8, and 9 from the design document, verifying
step sequencing, commentary emission, and pause/resume behaviour.

Uses hypothesis for property-based testing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from backend.app.engines.demo_orchestrator import (
    DEMO_STEPS,
    DemoOrchestrator,
    DemoState,
    SPEED_SETTINGS,
    STEP_COMMENTARY,
    STEP_HIGHLIGHT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MessageCollector:
    """Collects WebSocket messages emitted by the orchestrator."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, Any]]] = []

    async def callback(self, msg_type: str, payload: dict[str, Any]) -> None:
        self.messages.append((msg_type, payload))

    def get_by_type(self, msg_type: str) -> list[dict[str, Any]]:
        return [p for t, p in self.messages if t == msg_type]


def _make_orchestrator() -> DemoOrchestrator:
    """Create an orchestrator with zero pause for fast testing."""
    orch = DemoOrchestrator()
    orch._pause_duration_override = 0.0
    return orch


async def _run_demo(speed: str = "fast") -> tuple[DemoOrchestrator, MessageCollector]:
    """Run a demo to completion with zero pauses and return results."""
    orch = _make_orchestrator()
    collector = MessageCollector()

    await orch.start_demo(speed=speed, ws_callback=collector.callback)

    if orch._task:
        await asyncio.wait_for(orch._task, timeout=10.0)

    return orch, collector


# ---------------------------------------------------------------------------
# Property 7: Demo step sequence
# **Validates: Requirements 7.1**
# ---------------------------------------------------------------------------

class TestDemoStepSequence:
    """Property 7: For any demo run, steps execute in correct order:
    model_loading → dataset_preparation → training_configuration →
    training_execution → evaluation → text_generation.
    """

    @given(speed=st.sampled_from(["fast", "medium", "slow"]))
    @settings(max_examples=10, deadline=None)
    def test_steps_execute_in_correct_order(self, speed: str) -> None:
        """**Validates: Requirements 7.1**

        For any speed setting, the executed steps match the expected sequence.
        """
        loop = asyncio.new_event_loop()
        try:
            orch, _ = loop.run_until_complete(_run_demo(speed=speed))
            assert orch.state == DemoState.COMPLETED
            assert orch.executed_steps == DEMO_STEPS
        finally:
            loop.close()

    @given(speed=st.sampled_from(["fast", "medium", "slow"]))
    @settings(max_examples=10, deadline=None)
    def test_no_steps_skipped_or_reordered(self, speed: str) -> None:
        """**Validates: Requirements 7.1**

        For any demo run, step_start messages arrive in the correct order.
        """
        loop = asyncio.new_event_loop()
        try:
            _, collector = loop.run_until_complete(_run_demo(speed=speed))
            step_starts = [
                p["step"]
                for p in collector.get_by_type("demo_step")
                if p.get("action") == "step_start"
            ]
            assert step_starts == DEMO_STEPS
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Property 8: Demo commentary and highlight
# **Validates: Requirements 7.3, 7.4**
# ---------------------------------------------------------------------------

class TestDemoCommentaryAndHighlight:
    """Property 8: For any demo step, the Demo_Orchestrator emits a
    commentary message and the correct component is highlighted.
    """

    @given(speed=st.sampled_from(["fast", "medium", "slow"]))
    @settings(max_examples=10, deadline=None)
    def test_commentary_emitted_for_every_step(self, speed: str) -> None:
        """**Validates: Requirements 7.3**

        For any demo run, a commentary message is emitted for each step.
        """
        loop = asyncio.new_event_loop()
        try:
            _, collector = loop.run_until_complete(_run_demo(speed=speed))
            commentary_msgs = collector.get_by_type("commentary")
            commentary_steps = [m["step"] for m in commentary_msgs]
            for step in DEMO_STEPS:
                assert step in commentary_steps, f"Missing commentary for step '{step}'"
        finally:
            loop.close()

    @given(speed=st.sampled_from(["fast", "medium", "slow"]))
    @settings(max_examples=10, deadline=None)
    def test_correct_component_highlighted(self, speed: str) -> None:
        """**Validates: Requirements 7.4**

        For any demo step, the correct UI component is highlighted.
        """
        loop = asyncio.new_event_loop()
        try:
            _, collector = loop.run_until_complete(_run_demo(speed=speed))
            step_starts = [
                p for p in collector.get_by_type("demo_step")
                if p.get("action") == "step_start"
            ]
            for msg in step_starts:
                step = msg["step"]
                expected = STEP_HIGHLIGHT[step]
                assert msg["highlight"] == expected, (
                    f"Step '{step}': got '{msg['highlight']}', expected '{expected}'"
                )
        finally:
            loop.close()

    @given(speed=st.sampled_from(["fast", "medium", "slow"]))
    @settings(max_examples=10, deadline=None)
    def test_commentary_text_is_non_empty(self, speed: str) -> None:
        """**Validates: Requirements 7.3**

        For any demo step, the commentary text is non-empty.
        """
        loop = asyncio.new_event_loop()
        try:
            _, collector = loop.run_until_complete(_run_demo(speed=speed))
            commentary_msgs = collector.get_by_type("commentary")
            for msg in commentary_msgs:
                assert msg["text"], f"Empty commentary for step '{msg['step']}'"
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Property 9: Demo pause/resume round-trip
# **Validates: Requirements 7.5**
# ---------------------------------------------------------------------------

class TestDemoPauseResume:
    """Property 9: Pausing stops step progression, resuming continues
    from the exact same step without repeating or skipping.
    """

    @given(
        pause_after_step=st.integers(min_value=0, max_value=len(DEMO_STEPS) - 2),
    )
    @settings(max_examples=10, deadline=None)
    def test_pause_resume_continues_from_same_step(
        self, pause_after_step: int
    ) -> None:
        """**Validates: Requirements 7.5**

        Pausing after N steps and resuming completes all remaining steps.
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self._do_pause_resume_test(pause_after_step)
            )
        finally:
            loop.close()

    async def _do_pause_resume_test(self, pause_after_step: int) -> None:
        orch = _make_orchestrator()
        collector = MessageCollector()
        steps_executed: list[str] = []

        async def tracking_executor(step_name: str) -> None:
            steps_executed.append(step_name)
            if len(steps_executed) == pause_after_step + 1:
                await orch.pause()

        await orch.start_demo(
            speed="fast",
            ws_callback=collector.callback,
            step_executor=tracking_executor,
        )

        # Wait for pause
        for _ in range(100):
            await asyncio.sleep(0.01)
            if orch.state == DemoState.PAUSED:
                break

        assert orch.state == DemoState.PAUSED, "Demo did not pause"

        # Resume and wait for completion
        await orch.resume()
        if orch._task:
            await asyncio.wait_for(orch._task, timeout=10.0)

        assert orch.state == DemoState.COMPLETED
        assert orch.executed_steps == DEMO_STEPS

    @given(speed=st.sampled_from(["fast", "medium", "slow"]))
    @settings(max_examples=6, deadline=None)
    def test_pause_stops_progression(self, speed: str) -> None:
        """**Validates: Requirements 7.5**

        When paused, no new steps are executed.
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._do_pause_stops_test(speed))
        finally:
            loop.close()

    async def _do_pause_stops_test(self, speed: str) -> None:
        orch = _make_orchestrator()
        collector = MessageCollector()
        step_count = 0

        async def counting_executor(step_name: str) -> None:
            nonlocal step_count
            step_count += 1
            if step_count == 1:
                await orch.pause()

        await orch.start_demo(
            speed=speed,
            ws_callback=collector.callback,
            step_executor=counting_executor,
        )

        # Wait for pause
        for _ in range(100):
            await asyncio.sleep(0.01)
            if orch.state == DemoState.PAUSED:
                break

        if orch.state == DemoState.PAUSED:
            count_at_pause = step_count
            await asyncio.sleep(0.2)
            assert step_count == count_at_pause, "Steps continued while paused"

            # Clean up
            await orch.resume()
            if orch._task:
                await asyncio.wait_for(orch._task, timeout=10.0)
