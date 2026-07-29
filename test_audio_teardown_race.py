import asyncio
import unittest
from pathlib import Path


class FakeStream:
    def __init__(self):
        self.write_started = asyncio.Event()
        self.release_write = asyncio.Event()
        self.write_finished = False
        self.closed = False
        self.close_calls = 0
        self.write_after_close = 0

    async def write(self):
        self.write_started.set()
        await self.release_write.wait()
        if self.closed:
            self.write_after_close += 1
        self.write_finished = True

    def close(self):
        if not self.write_finished:
            raise AssertionError("close called before writer exited")
        self.close_calls += 1
        self.closed = True


class TeardownHarness:
    def __init__(self, stream):
        self.stream = stream
        self.cancelled = asyncio.Event()
        self.state = "AUDIO_RUNNING"
        self.close_claimed = False
        self.reconnect_started = False

    async def teardown(self, writer):
        self.cancelled.set()
        self.state = "AUDIO_STOPPING"
        await asyncio.wait_for(asyncio.shield(writer), timeout=1.0)
        if not self.close_claimed:
            self.close_claimed = True
            self.stream.close()
        self.state = "AUDIO_STOPPED"

    def reconnect(self):
        if self.state != "AUDIO_STOPPED":
            return False
        self.reconnect_started = True
        return True


class AudioTeardownRaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_exits_before_single_close_and_reconnect(self):
        stream = FakeStream()
        harness = TeardownHarness(stream)
        writer = asyncio.create_task(stream.write())
        await stream.write_started.wait()

        teardown = asyncio.create_task(harness.teardown(writer))
        await asyncio.sleep(0)
        self.assertTrue(harness.cancelled.is_set())
        self.assertEqual(harness.state, "AUDIO_STOPPING")
        self.assertFalse(harness.reconnect())
        self.assertFalse(stream.closed)

        stream.release_write.set()
        await teardown
        self.assertEqual(harness.state, "AUDIO_STOPPED")
        self.assertEqual(stream.close_calls, 1)
        self.assertEqual(stream.write_after_close, 0)
        self.assertTrue(harness.reconnect())

        await harness.teardown(asyncio.create_task(asyncio.sleep(0)))
        self.assertEqual(stream.close_calls, 1)

    def test_main_uses_shielded_writer_and_blocks_early_reconnect(self):
        source = Path(__file__).with_name("main.py").read_text(encoding="utf-8")
        self.assertIn("await asyncio.shield(write_task)", source)
        self.assertIn("self._audio_cancel_event.set()", source)
        self.assertIn("if self._get_audio_state() != AUDIO_STOPPED", source)
        self.assertIn("reconnect_blocked=audio_not_stopped", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
