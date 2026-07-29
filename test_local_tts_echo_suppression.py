import asyncio
import contextlib
import io
import unittest
from unittest.mock import patch

from actions.windows_action_registry import WINDOWS_ACTION_REGISTRY
from main import JarvisLive


def tearDownModule():
    WINDOWS_ACTION_REGISTRY.shutdown(wait=False)


class FakeUI:
    muted = False
    on_text_command = None
    on_mute_changed = None

    def set_state(self, state):
        pass


class LocalTtsEchoSuppressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_is_cleared_before_and_after_and_gate_reopens_once(self):
        jarvis = JarvisLive(FakeUI(), object())
        jarvis.out_queue = asyncio.Queue()
        jarvis.audio_in_queue = asyncio.Queue()
        jarvis.out_queue.put_nowait({"data": b"old"})
        jarvis._input_transcript_parts[:] = ["old transcript"]
        item = {
            "action": "list_browser_tabs", "text": "Bạn đang mở 2 tab.",
            "success": True, "status": "completed",
        }
        output = io.StringIO()
        with patch("main.shutil.which", return_value="powershell.exe"), patch(
            "main.LOCAL_TTS_RUNNER.speak",
            return_value={"success": True, "status": "completed"},
        ), contextlib.redirect_stdout(output):
            await jarvis._speak_fast_response(item)
        logs = output.getvalue()
        self.assertTrue(jarvis.out_queue.empty())
        self.assertEqual(jarvis._input_transcript_parts, [])
        self.assertFalse(jarvis._local_echo_guard_active())
        self.assertEqual(logs.count("mic_queue_cleared_before=true"), 1)
        self.assertEqual(logs.count("mic_queue_cleared_after=true"), 1)
        self.assertEqual(logs.count("gate_reopened=true"), 1)

    def test_echo_transcript_is_discarded_without_new_turn(self):
        jarvis = JarvisLive(FakeUI(), object())
        jarvis._session_generation = 1
        jarvis._route_turn_id = 1
        jarvis._local_tts_active = True
        jarvis._stale_response_turns.add((1, 1))
        initial_turn = jarvis._voice_turn_sequence
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(jarvis._discard_echo_transcript_if_needed())
            self.assertTrue(jarvis._discard_echo_transcript_if_needed())
        self.assertEqual(jarvis._voice_turn_sequence, initial_turn)
        self.assertEqual(output.getvalue().count("transcript_discarded"), 1)
        self.assertTrue(jarvis._discard_late_gemini_audio(b"audio"))
        self.assertEqual(jarvis._voice_turn_sequence, initial_turn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
