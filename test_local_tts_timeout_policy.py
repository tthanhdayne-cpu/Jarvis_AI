import subprocess
import unittest

from actions.local_tts import (
    HARD_SPEECH_CAP, LocalTTSRunner, speech_timeout_for,
)


class FakeStdin:
    def __init__(self):
        self.text = ""
        self.closed = False

    def write(self, text):
        self.text += text

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, required=0.0, timeout=False):
        self.stdin = FakeStdin()
        self.stderr = iter(["JARVIS_TTS_READY|VmlldG5hbWVzZSBWb2ljZQ==|vi-VN|False\n"])
        self.required = required
        self.force_timeout = timeout
        self.done = False
        self.terminated = False
        self.killed = False
        self.wait_timeouts = []

    def poll(self):
        return 0 if self.done else None

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.force_timeout and not self.terminated:
            raise subprocess.TimeoutExpired("tts", timeout)
        self.done = True
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class LocalTtsTimeoutPolicyTests(unittest.TestCase):
    def test_short_sentence_completes(self):
        process = FakeProcess()
        runner = LocalTTSRunner(lambda *a, **k: process, logger=lambda _: None)
        result = runner.speak("Bạn đang mở 4 tab.", "powershell.exe")
        self.assertTrue(result["success"])
        self.assertTrue(process.stdin.closed)

    def test_long_sentence_gets_more_than_four_seconds(self):
        text = "Tên tab tiếng Việt. " * 8
        self.assertGreater(speech_timeout_for(text), 4.0)
        self.assertLessEqual(speech_timeout_for(text), HARD_SPEECH_CAP)

    def test_timeout_cleans_process_without_fallback(self):
        process = FakeProcess(timeout=True)
        runner = LocalTTSRunner(lambda *a, **k: process, logger=lambda _: None)
        result = runner.speak("Một câu local.", "powershell.exe")
        self.assertEqual(result["status"], "local_tts_timeout")
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertNotIn("fallback", result.get("status", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
