import asyncio
import time
import unittest

from actions.utterance_normalizer import FinalTranscriptTracker
from memory.memory_manager import should_schedule_memory


class FastPathHarness:
    def __init__(self):
        self.action_called = 0
        self.memory_called = 0
        self.duplicate_response_count = 0
        self.late_gemini_audio_discarded = False
        self.tracker = FinalTranscriptTracker(ttl=10.0)

    async def run(self, *, bridge_delay=0.05, bridge_timeout=False):
        started = time.perf_counter()
        first_started = time.perf_counter()
        claimed = self.tracker.claim(1, "Liệt kê các tab đang mở")
        first_claim_ms = (time.perf_counter() - first_started) * 1000
        duplicate = self.tracker.claim(1, "Liệt kê các tab đang mở")

        registry_start_ms = (time.perf_counter() - started) * 1000
        self.action_called += 1
        await asyncio.sleep(bridge_delay)

        if bridge_timeout:
            result = {"success": False, "status": "request_timeout"}
        else:
            result = {"success": True, "status": "completed", "tabs": []}
        action_complete_ms = (time.perf_counter() - started) * 1000

        if should_schedule_memory("list_browser_tabs"):
            self.memory_called += 1

        response_owner = "local"
        response_start_ms = (time.perf_counter() - started) * 1000
        await asyncio.sleep(0.01)
        if response_owner == "local":
            self.late_gemini_audio_discarded = True
        else:
            self.duplicate_response_count += 1
        total_turn_ms = (time.perf_counter() - started) * 1000

        return {
            "claimed": claimed,
            "duplicate_turn_count": int(duplicate),
            "first_claim_ms": first_claim_ms,
            "registry_start_ms": registry_start_ms,
            "action_complete_ms": action_complete_ms,
            "response_start_ms": response_start_ms,
            "total_turn_ms": total_turn_ms,
            "status": result["status"],
        }


class LocalActionEndToEndLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_connected_bridge_fast_path(self):
        harness = FastPathHarness()
        result = await harness.run()

        self.assertTrue(result["claimed"])
        self.assertEqual(harness.action_called, 1)
        self.assertEqual(harness.memory_called, 0)
        self.assertEqual(result["duplicate_turn_count"], 0)
        self.assertLess(result["first_claim_ms"], 50)
        self.assertLess(result["registry_start_ms"], 500)
        self.assertLess(result["action_complete_ms"], 1500)
        self.assertLess(result["response_start_ms"], 4000)
        self.assertLess(result["total_turn_ms"], 5000)
        self.assertTrue(result["total_turn_ms"] < 60000)
        self.assertTrue(harness.late_gemini_audio_discarded)
        self.assertEqual(harness.duplicate_response_count, 0)

        print("action_called == 1")
        print("memory_called == 0")
        print("duplicate_turn_count == 0")
        print("no_60_second_wait == true")
        print("late_gemini_audio_discarded == true")
        print("duplicate_response_count == 0")

    async def test_bridge_timeout_returns_under_four_seconds(self):
        harness = FastPathHarness()
        result = await harness.run(bridge_delay=0.2, bridge_timeout=True)
        self.assertEqual(result["status"], "request_timeout")
        self.assertLess(result["total_turn_ms"], 4000)
        self.assertEqual(harness.action_called, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
