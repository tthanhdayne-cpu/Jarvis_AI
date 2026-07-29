import contextlib
import io
import unittest

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


class LocalOwnerCleanupTests(unittest.TestCase):
    def test_stale_audio_has_one_start_and_one_summary(self):
        jarvis = JarvisLive(FakeUI(), object())
        jarvis._session_generation = 3
        jarvis._route_turn_id = 9
        jarvis._route_started_at = 10.0
        jarvis._stale_response_turns.add((3, 9))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for chunk in (b"aa", b"bbb", b"c"):
                self.assertTrue(jarvis._discard_late_gemini_audio(chunk))
            self.assertTrue(jarvis._handle_server_turn_complete())
            self.assertTrue(jarvis._handle_server_turn_complete())
        logs = output.getvalue()
        self.assertEqual(logs.count("late_gemini_audio_discard_started=true"), 1)
        self.assertEqual(logs.count("late_gemini_audio_discard_summary"), 1)
        self.assertIn("chunks=3 bytes=6", logs)
        self.assertEqual(logs.count("server_turn_complete_cleanup=true"), 1)
        self.assertNotIn("stage=turn_complete", logs)

    def test_latency_stage_and_owner_release_are_logged_once(self):
        jarvis = JarvisLive(FakeUI(), object())
        jarvis._session_generation = 1
        jarvis._route_turn_id = 4
        jarvis._route_started_at = 1.0
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            jarvis._latency_log("response_owner_released")
            jarvis._latency_log("response_owner_released")
        self.assertEqual(output.getvalue().count("stage=response_owner_released"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
