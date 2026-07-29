import ast
import unittest
from pathlib import Path


MAIN_PATH = Path(__file__).with_name("main.py")
SOURCE = MAIN_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Missing constant: {name}")


def method_source(name):
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(SOURCE, node) or ""
    raise AssertionError(f"Missing method: {name}")


class LiveAudioConfigTests(unittest.TestCase):
    def test_native_audio_model_and_rates(self):
        self.assertEqual(
            constant("LIVE_MODEL"),
            "models/gemini-2.5-flash-native-audio-preview-12-2025",
        )
        self.assertEqual(constant("SEND_SAMPLE_RATE"), 16000)
        self.assertEqual(constant("RECEIVE_SAMPLE_RATE"), 24000)

    def test_live_config_remains_audio_with_search_and_functions(self):
        build = method_source("_build_config")
        self.assertIn('response_modalities=["AUDIO"]', build)
        self.assertIn("speech_config=types.SpeechConfig", build)
        self.assertIn('{"google_search": {}}', build)
        self.assertIn('{"function_declarations": TOOL_DECLARATIONS}', build)

    def test_input_audio_mime_is_explicit_pcm_16khz(self):
        callback = method_source("_listen_audio")
        self.assertIn('"mime_type": "audio/pcm;rate=16000"', callback)
        self.assertNotIn('"mime_type": "audio/pcm"', callback)

    def test_local_tts_does_not_mutate_or_send_audio_to_live(self):
        local_tts = method_source("_speak_fast_response")
        self.assertNotIn("LiveConnectConfig", local_tts)
        self.assertNotIn("send_realtime_input", local_tts)
        self.assertNotIn("audio_in_queue", local_tts)

    def test_reconnect_rebuilds_same_validated_config(self):
        active = method_source("_run_active_session")
        self.assertIn("config = self._build_config()", active)
        self.assertIn("self._validate_live_audio_config(config)", active)
        self.assertIn("client.aio.live.connect(model=LIVE_MODEL, config=config)", active)


if __name__ == "__main__":
    unittest.main(verbosity=2)
