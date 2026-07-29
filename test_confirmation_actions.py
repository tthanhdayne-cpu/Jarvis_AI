"""Independent tests for voice-only close-tab confirmation; no real keys sent."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from actions.confirmation_manager import ConfirmationManager
from actions.windows_browser_actions import ForegroundWindow


BROWSER = ForegroundWindow(101, 202, "chrome.exe", "YouTube - Chrome", True)


class Runtime:
    def __init__(self):
        self.state = "ACTIVE"
        self.sleep = False
        self.generation = 7
        self.event = threading.Event()
        self.state_getter = lambda: self.state
        self.sleep_intent_getter = lambda: self.sleep
        self.generation_getter = lambda: self.generation
        self.cancellation_event = self.event


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.sent = []
        self.manager = ConfirmationManager(
            sender=lambda snapshot: self.sent.append(snapshot) or {"message": "closed", "data": {}},
            clock=lambda: self.now[0],
        )
        self.runtime = Runtime()
        self.patches = [
            patch("actions.confirmation_manager.require_foreground_browser", return_value=BROWSER),
            patch("actions.confirmation_manager.inspect_foreground_window", return_value=BROWSER),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def request(self):
        result = self.manager.request_close_browser_tab(session_generation=7, source_turn=10)
        self.assertEqual(result["status"], "confirmation_required")
        self.assertTrue(result["success"])
        self.assertEqual(
            result["data"]["speak_exactly"],
            "Bạn có chắc muốn đóng tab hiện tại không?",
        )
        self.assertEqual(self.sent, [])

    def test_request_does_not_send(self):
        self.request()

    def test_same_turn_cannot_approve_and_pending_survives(self):
        self.request()
        self.assertIsNone(
            self.manager.resolve_voice_transcript("Xác nhận", turn_id=10, runtime=self.runtime)
        )
        self.assertIsNotNone(self.manager.pending)
        self.assertEqual(self.sent, [])

    def test_unrelated_next_turn_keeps_pending(self):
        self.request()
        self.assertIsNone(
            self.manager.resolve_voice_transcript("Mở Gmail", turn_id=11, runtime=self.runtime)
        )
        self.assertIsNotNone(self.manager.pending)
        self.assertEqual(self.sent, [])

    def test_approve_sends_once(self):
        self.request()
        result = self.manager.resolve_voice_transcript("Xác nhận", turn_id=11, runtime=self.runtime)
        self.assertTrue(result["success"])
        self.assertEqual(len(self.sent), 1)
        self.assertIsNone(self.manager.resolve_voice_transcript("Yes", turn_id=12, runtime=self.runtime))
        self.assertEqual(len(self.sent), 1)

    def test_cancel_does_not_send(self):
        self.request()
        result = self.manager.resolve_voice_transcript("Không", turn_id=11, runtime=self.runtime)
        self.assertEqual(result["status"], "cancelled_user")
        self.assertEqual(self.sent, [])

    def test_expiry(self):
        self.request()
        self.now[0] = 131.0
        result = self.manager.resolve_voice_transcript("Confirm", turn_id=11, runtime=self.runtime)
        self.assertEqual(result["status"], "confirmation_expired")
        self.assertEqual(self.sent, [])

    def test_sleep_cancellation(self):
        self.request()
        self.manager.cancel("cancelled_sleep")
        self.assertIsNone(self.manager.resolve_voice_transcript("Yes", turn_id=11, runtime=self.runtime))
        self.assertEqual(self.sent, [])

    def test_session_mismatch(self):
        self.request()
        self.runtime.generation = 8
        result = self.manager.resolve_voice_transcript("Yes", turn_id=11, runtime=self.runtime)
        self.assertEqual(result["status"], "stale_session")

    def test_foreground_changed(self):
        self.request()
        changed = ForegroundWindow(999, 202, "chrome.exe", "Other", True)
        with patch("actions.confirmation_manager.inspect_foreground_window", return_value=changed):
            result = self.manager.resolve_voice_transcript("Yes", turn_id=11, runtime=self.runtime)
        self.assertEqual(result["status"], "browser_not_foreground")
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
