import asyncio
import unittest
from unittest.mock import Mock, patch

from actions.microphone_state import MICROPHONE_STATE
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


class LocalRoutingMuteIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        MICROPHONE_STATE.set_enabled(True)
        self.jarvis = JarvisLive(FakeUI(), object())
        self.jarvis._state = "ACTIVE"
        self.jarvis._session_generation = 1
        self.jarvis._route_turn_id = 1
        self.jarvis.audio_in_queue = asyncio.Queue()
        self.jarvis.out_queue = asyncio.Queue()
        self.jarvis._start_fast_response_owner = Mock()
        self.dispatch = patch(
            "main.WINDOWS_ACTION_REGISTRY.dispatch",
            return_value={
                "success": True,
                "status": "completed",
                "data": {"tabs": [{"title": "YouTube"}, {"title": "Gmail"}]},
            },
        )
        self.dispatch_mock = self.dispatch.start()

    async def asyncTearDown(self):
        self.dispatch.stop()
        MICROPHONE_STATE.set_enabled(True)

    async def test_unmuted_list_tabs_claims_once_without_gemini_owner(self):
        claimed = await self.jarvis._claim_local_intent(
            "Liệt kê các tab đang mở"
        )
        duplicate = await self.jarvis._claim_local_intent(
            "Liệt kê các tab đang mở"
        )
        self.assertTrue(claimed)
        self.assertTrue(duplicate)
        self.assertEqual(self.dispatch_mock.call_count, 1)
        self.assertEqual(self.dispatch_mock.call_args.args[:2], ("list_browser_tabs", {}))
        self.assertEqual(
            self.jarvis._response_owner,
            "action_pending",
            "the mocked selector has not yet chosen vi-VN local TTS or Gemini",
        )
        self.assertEqual(self.jarvis._start_fast_response_owner.call_count, 1)

    async def test_muted_audio_or_transcript_cannot_route(self):
        MICROPHONE_STATE.set_enabled(False)
        claimed = await self.jarvis._claim_local_intent(
            "Liệt kê các tab đang mở"
        )
        self.assertFalse(claimed)
        self.dispatch_mock.assert_not_called()

    async def test_local_tts_echo_cannot_create_another_turn(self):
        self.jarvis._local_tts_active = True
        claimed = await self.jarvis._claim_local_intent(
            "Liệt kê các tab đang mở"
        )
        self.assertFalse(claimed)
        self.dispatch_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
