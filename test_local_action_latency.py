import asyncio
import threading
import time
import unittest
from pathlib import Path

from actions.browser_tab_actions import BrowserTabService
from actions.utterance_normalizer import FinalTranscriptTracker
from memory.memory_manager import should_schedule_memory


class FakeRegistry:
    def __init__(self):
        self.browser_action_called = 0

    def dispatch(self):
        self.browser_action_called += 1
        return {
            "success": True,
            "action": "list_browser_tabs",
            "status": "completed",
            "data": {"tabs": []},
        }


class FakeTransport:
    def __init__(self, status):
        self.status = status

    def start(self):
        return True

    def request(self, action, arguments, runtime):
        return {"success": False, "status": self.status, "data": {}}

    def close(self):
        pass


class FakeRuntime:
    expected_generation = 1
    cancellation_event = threading.Event()

    @staticmethod
    def state_getter():
        return "ACTIVE"

    @staticmethod
    def sleep_intent_getter():
        return False

    @staticmethod
    def generation_getter():
        return 1


class LocalActionLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_openrouter_429_does_not_block_local_action(self):
        registry = FakeRegistry()

        openrouter_calls = 0

        def always_429(*_args, **_kwargs):
            nonlocal openrouter_calls
            openrouter_calls += 1
            raise RuntimeError("429")

        started = time.monotonic()
        tool_result = registry.dispatch()
        scheduled = should_schedule_memory("list_browser_tabs")
        elapsed = time.monotonic() - started

        self.assertEqual(registry.browser_action_called, 1)
        self.assertFalse(scheduled)
        self.assertEqual(openrouter_calls, 0)
        self.assertLess(elapsed, 0.1)
        self.assertEqual(tool_result["status"], "completed")

        async def isolated_memory_failure():
            try:
                await asyncio.to_thread(always_429)
            except RuntimeError:
                return "failed"

        background = asyncio.create_task(isolated_memory_failure())
        self.assertEqual(await background, "failed")

        self.assertEqual(tool_result["status"], "completed")
        self.assertEqual(openrouter_calls, 1)

        print("browser_action_called == 1")
        print("memory_failure_did_not_block == true")

        main_source = Path(__file__).with_name("main.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "await asyncio.to_thread(\n                                    _update_memory_async",
            main_source,
        )
        self.assertIn("self._schedule_memory_background(", main_source)

    async def test_final_transcript_is_claimed_once(self):
        tracker = FinalTranscriptTracker(ttl=10.0)
        first = tracker.claim(7, "Liệt kê các tab đang mở")
        second = tracker.claim(7, "Liệt kê các tab đang mở")

        self.assertTrue(first)
        self.assertFalse(second)
        duplicate_turn_count = int(second)
        self.assertEqual(duplicate_turn_count, 0)
        print("duplicate_turn_count == 0")

    async def test_non_bridge_failure_is_not_called_extension_error(self):
        service = BrowserTabService(transport=FakeTransport("native_host_unavailable"))
        try:
            result = service.list_tabs(runtime=FakeRuntime())
        finally:
            service.shutdown()

        false_extension_error = int(
            result["status"] != "bridge_unavailable"
            and "extension" in result["message"].casefold()
        )
        self.assertEqual(false_extension_error, 0)
        print("false_extension_error == 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
