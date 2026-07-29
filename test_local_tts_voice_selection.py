import unittest
from pathlib import Path

from actions.local_tts import LocalTTSRunner, choose_voice


class LocalTtsVoiceSelectionTests(unittest.TestCase):
    def test_vi_vn_is_preferred(self):
        selected = choose_voice([
            {"name": "English Voice", "culture": "en-US"},
            {"name": "Vietnamese Voice", "culture": "vi-VN"},
        ])
        self.assertEqual(selected["name"], "Vietnamese Voice")
        self.assertFalse(selected["fallback_used"])

    def test_fallback_is_explicit(self):
        selected = choose_voice([{"name": "Fallback", "culture": "en-US"}])
        self.assertTrue(selected["fallback_used"])
        self.assertEqual(selected["culture"], "en-US")

    def test_unicode_and_no_shell_true(self):
        source = LocalTTSRunner._script()
        self.assertIn("InputEncoding", source)
        self.assertIn("vi-VN", source)
        self.assertNotIn("shell=True", source)
        module_source = Path("actions/local_tts.py").read_text(encoding="utf-8")
        self.assertIn("shell=False", module_source)
        text = "Bạn đang mở năm tab."
        self.assertEqual(text.encode("utf-8").decode("utf-8"), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
