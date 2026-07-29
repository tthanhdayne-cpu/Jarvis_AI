import asyncio
import unittest

from actions.microphone_state import MICROPHONE_STATE, MicrophoneState
from main import JarvisLive
from actions.windows_action_registry import WINDOWS_ACTION_REGISTRY


def tearDownModule():
    WINDOWS_ACTION_REGISTRY.shutdown(wait=False)


class FakeUI:
    muted = False
    on_text_command = None
    on_mute_changed = None

    def set_state(self, state):
        pass


class FakeSession:
    def __init__(self):
        self.sent = []

    async def send_realtime_input(self, media):
        self.sent.append(media)


class MicrophoneMuteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        MICROPHONE_STATE.set_enabled(True)

    async def asyncTearDown(self):
        MICROPHONE_STATE.set_enabled(True)

    def test_thread_safe_state_persists_until_explicit_unmute(self):
        state = MicrophoneState(enabled=True)
        self.assertTrue(state.set_enabled(False))
        self.assertTrue(state.muted)
        self.assertTrue(state.muted)  # reconnect/turn transitions do not reset it
        self.assertTrue(state.set_enabled(True))
        self.assertTrue(state.enabled)

    async def test_sender_drops_stale_chunk_while_muted(self):
        jarvis = JarvisLive(FakeUI(), object())
        jarvis._state = "ACTIVE"
        jarvis._accept_gemini_input = True
        jarvis._sleep_requested = asyncio.Event()
        jarvis.out_queue = asyncio.Queue()
        jarvis.session = FakeSession()
        MICROPHONE_STATE.set_enabled(False)
        jarvis.out_queue.put_nowait({"data": b"old"})
        task = asyncio.create_task(jarvis._send_realtime())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(jarvis.session.sent, [])

    def test_callback_queue_rejects_audio_while_muted(self):
        jarvis = JarvisLive(FakeUI(), object())
        jarvis._state = "ACTIVE"
        jarvis._mic_owner = "gemini"
        jarvis._accept_gemini_input = True
        jarvis._mic_generation = 3
        jarvis._sleep_requested = None
        jarvis.out_queue = asyncio.Queue()
        MICROPHONE_STATE.set_enabled(False)
        jarvis._queue_mic_chunk({"data": b"new"}, 3)
        self.assertTrue(jarvis.out_queue.empty())


if __name__ == "__main__":
    unittest.main(verbosity=2)
