import asyncio
import contextlib
import io
import time
import unittest
from pathlib import Path
from unittest import mock

import main
from actions.microphone_state import MICROPHONE_STATE
from actions.windows_action_registry import WINDOWS_ACTION_REGISTRY


def tearDownModule():
    WINDOWS_ACTION_REGISTRY.shutdown(wait=False)


class FakeUI:
    def __init__(self):
        self.on_text_command = None
        self.on_mute_changed = None
        self.states = []
        self.logs = []

    @property
    def muted(self):
        return MICROPHONE_STATE.muted

    def set_state(self, state):
        self.states.append(state)

    def write_log(self, text):
        self.logs.append(text)


class FakeSession:
    def __init__(self):
        self.client_turns = []

    async def send_client_content(self, **kwargs):
        self.client_turns.append(kwargs)


class FakeBridge:
    def __init__(self):
        self.request_called = 0

    def list_tabs(self):
        self.request_called += 1
        return {
            "success": True,
            "status": "completed",
            "data": {
                "tabs": [
                    {"title": "YouTube", "url": "https://youtube.com/watch?v=secret"},
                    {"title": "Gmail", "url": "https://mail.google.com/mail/u/0"},
                    {"title": "GitHub", "url": "https://github.com/private/repo"},
                    {"title": "ChatGPT", "url": "https://chatgpt.com/c/secret"},
                ]
            },
        }


class PhaseBStabilizationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        MICROPHONE_STATE.set_enabled(True)
        self.ui = FakeUI()
        self.jarvis = main.JarvisLive(self.ui, object())
        self.jarvis._state = "ACTIVE"
        self.jarvis._session_generation = 7
        self.jarvis.session = FakeSession()
        self.jarvis.audio_in_queue = asyncio.Queue()
        self.jarvis.out_queue = asyncio.Queue()
        self.jarvis._local_voice_policy = {
            "use_local_tts": False,
            "requested_culture": "vi-VN",
            "selected_name": "Microsoft David Desktop",
            "selected_culture": "en-US",
            "fallback_used": True,
        }

    async def asyncTearDown(self):
        tasks = list(self.jarvis._local_response_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        MICROPHONE_STATE.set_enabled(True)

    async def test_local_list_tabs_has_one_registry_bridge_and_response_owner(self):
        bridge = FakeBridge()
        registry_called = 0

        def dispatch(name, args, runtime):
            nonlocal registry_called
            registry_called += 1
            self.assertEqual(name, "list_browser_tabs")
            return bridge.list_tabs()

        output = io.StringIO()
        with mock.patch.object(
            main.WINDOWS_ACTION_REGISTRY, "dispatch", side_effect=dispatch
        ), mock.patch.object(main.LOCAL_TTS_RUNNER, "speak") as local_speak, \
                contextlib.redirect_stdout(output):
            self.assertTrue(
                self.jarvis._begin_latency_turn("Liệt kê các tab đang mở")
            )
            original_turn_count = self.jarvis._voice_turn_sequence
            claimed = await self.jarvis._claim_local_intent(
                "Liệt kê các tab đang mở"
            )
            await asyncio.sleep(0)

        logs = output.getvalue()
        prompt = self.jarvis.session.client_turns[0]["turns"]["parts"][0]["text"]
        self.assertTrue(claimed)
        self.assertEqual(registry_called, 1)
        self.assertEqual(bridge.request_called, 1)
        self.assertIn("registry_status=completed", logs)
        self.assertIn("Bạn đang mở 4 tab", prompt)
        for title in ("YouTube", "Gmail", "GitHub", "ChatGPT"):
            self.assertIn(title, prompt)
        self.assertNotIn("watch?v=secret", prompt)
        self.assertEqual(self.jarvis._response_owner, "gemini_action")
        self.assertTrue(self.jarvis._gemini_action_waiting)
        self.assertEqual(len(self.jarvis.session.client_turns), 1)
        self.assertEqual(self.jarvis._voice_turn_sequence, original_turn_count)
        local_speak.assert_not_called()
        self.assertIn("[JARVIS MEMORY] skipped_reason=local_action", logs)
        self.assertIn("[JARVIS SEARCH] skipped_reason=local_action", logs)
        self.assertNotIn("bridge_unavailable", logs)

        item = {
            "turn_id": self.jarvis._route_turn_id,
            "session_generation": self.jarvis._session_generation,
            "action": "list_browser_tabs",
            "success": True,
        }
        self.assertTrue(self.jarvis._release_action_owner(item, "test_complete"))
        self.assertFalse(self.jarvis._release_action_owner(item, "duplicate"))

    async def test_muted_transcript_cannot_reach_registry(self):
        MICROPHONE_STATE.set_enabled(False)
        with mock.patch.object(
            main.WINDOWS_ACTION_REGISTRY, "dispatch"
        ) as dispatch:
            claimed = await self.jarvis._claim_local_intent(
                "Liệt kê các tab đang mở"
            )
        self.assertFalse(claimed)
        dispatch.assert_not_called()
        self.assertEqual(self.jarvis._voice_turn_sequence, 0)

    async def test_english_only_policy_uses_finite_ui_fallback_not_local_tts(self):
        item = {
            "turn_id": 1,
            "session_generation": 7,
            "action": "list_browser_tabs",
            "status": "completed",
            "success": True,
            "text": "Bạn đang mở 4 tab: YouTube, Gmail, GitHub và ChatGPT.",
            "needs_prompt": False,
        }
        self.jarvis._route_turn_id = 1
        self.jarvis._route_started_at = time.perf_counter()
        self.jarvis._pending_fast_response = item
        with mock.patch.object(main, "GEMINI_ACTION_FIRST_AUDIO_DEADLINE", 0.01), \
                mock.patch.object(main.LOCAL_TTS_RUNNER, "speak") as local_speak:
            self.jarvis._start_fast_response_owner()
            await asyncio.sleep(0.03)
        local_speak.assert_not_called()
        self.assertIsNone(self.jarvis._response_owner)
        self.assertEqual(self.ui.logs, [f"Jarvis: {item['text']}"])
        self.assertEqual(
            len(self.jarvis._response_release_keys), 1,
            "deadline must release the owner exactly once",
        )

    def test_canonical_sources_do_not_restore_removed_parallel_paths(self):
        source = Path("main.py").read_text(encoding="utf-8")
        ui_source = Path("ui.py").read_text(encoding="utf-8")
        native_source = Path("native_host/jarvis_native_host.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("def _detect_obvious_local_intent", source)
        self.assertNotIn("self._muted", ui_source)
        self.assertIn("toggle_source={source}", ui_source)
        self.assertIn("except EOFError", native_source)
        self.assertIn('_mark_disconnected(connection, "eof")', native_source)
        self.assertNotIn("traceback.print_exc", native_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
