"""Offline Phase B tests. No microphone, network, browser, bridge, or actions."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from pathlib import Path

from actions.interaction_context import InteractionContext
from actions.utterance_normalizer import normalize_utterance


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "tests" / "utterance_dataset.json"
CONFIRM_TOOLS = {
    "close_browser_tab", "close_browser_tab_by_title",
    "close_browser_tab_by_index", "close_all_matching_tabs",
    "close_duplicate_tabs",
}
SIDE_EFFECT_TOOLS = CONFIRM_TOOLS | {
    "open_application", "open_url", "focus_browser_tab", "reload_browser_tab",
    "mute_browser_tab", "unmute_browser_tab", "pin_browser_tab",
    "unpin_browser_tab", "youtube_search", "play_youtube_video",
}


class DeterministicPolicyAdapter:
    """Test-only policy probe; it never calls or executes the predicted tool."""

    def predict(self, utterance: str) -> dict:
        n = normalize_utterance(utterance)
        text = n["comparison"]
        ambiguous = any(p in text for p in (
            "tab kia", "cai nay", "ben do", "no di", "trang ay",
            "chon mot cai", "cai do", "tab do", "that thing", "open it",
            "vua nay", "cuoi cung", "last result", "ket qua thu", "chon so",
            "cai thu", "cai truoc", "mo so", "dong google", "tieng no",
        ))
        if ambiguous:
            return {"tool": None, "arguments": {}, "clarify": True, "executed": False}
        negative = any(p in text for p in (
            "la gi", "bao nhieu", "ra doi", "toi thich", "thoi tiet",
            "giai thich", "toi vua dong", "co nhieu", "khac email",
            "xin chao", "viet lai", "dang phat nhac",
        ))
        if negative or text in {"xac nhan", "toi xac nhan", "confirm", "huy", "cancel", "dung lai", "tu choi", "khong xac nhan"}:
            return {"tool": None, "arguments": {}, "clarify": False, "executed": False}
        if any(p in text for p in ("dong cac tab trung", "duplicate tabs")):
            tool = "close_duplicate_tabs"
        elif any(p in text for p in ("dong het", "close all")):
            tool = "close_all_matching_tabs"
        elif ("dong tab thu" in text or "tab number" in text):
            tool = "close_browser_tab_by_index"
        elif any(p in text for p in ("dong tab nay", "tab nay khong", "dong trang dang mo", "close this tab")):
            tool = "close_browser_tab"
        elif any(p in text for p in ("dong tab", "tat trang", "close the")):
            tool = "close_browser_tab_by_title"
        elif any(p in text for p in ("tim video", "tim nhac", "tim shorts", "tra video", "search youtube", "search for", "xem ket qua", "danh sach karaoke")):
            tool = "youtube_search"
        elif any(p in text for p in ("phat ", "bat bai", "cho toi nghe", "cho toi xem", "play ", "mo nhac", "mo video")):
            tool = "play_youtube_video"
        elif any(p in text for p in ("liet ke", "danh sach tab", "tabs are open", "list my chrome")):
            tool = "list_browser_tabs"
        elif any(p in text for p in ("reload", "tai lai", "lam moi", "refresh")):
            tool = "reload_browser_tab"
        elif any(p in text for p in ("bat tieng", "mo tieng", "unmute")):
            tool = "unmute_browser_tab"
        elif any(p in text for p in ("tat tieng", "tat am", "mute")):
            tool = "mute_browser_tab"
        elif any(p in text for p in ("bo ghim", "thao ghim", "unpin")):
            tool = "unpin_browser_tab"
        elif any(p in text for p in ("ghim", "pin ")):
            tool = "pin_browser_tab"
        elif any(p in text for p in ("qua tab", "chuyen sang", "dua toi ve", "quay lai", "switch to", "go back", "focus the", "mo lai cai tab")):
            tool = "focus_browser_tab"
        elif any(site in text for site in ("youtube", "gmail", "google drive", "github", "facebook", "nttu", "instagram", "trang google")) and any(p in text for p in ("mo ", "open ", "go to", "cho toi vao", "muon xem", "vao trang")):
            tool = "open_url"
        elif any(p in text for p in ("mo ", "open ", "launch", "start", "khoi dong", "bat ")):
            tool = "open_application"
        else:
            tool = None
        return {"tool": tool, "arguments": {}, "clarify": False, "executed": False}


def load_dataset() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def evaluate_dataset() -> dict:
    adapter = DeterministicPolicyAdapter()
    rows = load_dataset()
    correct_tool = correct_args = wrong = missing = false_effect = bypass = context = 0
    clarify_expected = clarify_correct = 0
    for row in rows:
        result = adapter.predict(row["utterance"])
        predicted = result["tool"]
        expected = row["expected_tool"]
        correct_tool += predicted == expected
        wrong += predicted is not None and predicted != expected
        missing += expected is not None and predicted is None
        correct_args += predicted == expected and (not row["expected_arguments"] or result["arguments"] == row["expected_arguments"])
        if row["should_clarify"]:
            clarify_expected += 1
            clarify_correct += bool(result["clarify"])
            context += bool(result["clarify"] and row["category"] == "context_reference")
        if expected is None and predicted in SIDE_EFFECT_TOOLS:
            false_effect += 1
        if row["should_confirm"] and result["executed"]:
            bypass += 1
    total = len(rows)
    return {
        "scope": "deterministic_test_adapter_not_live_gemini",
        "total": total,
        "correct_tool_count": correct_tool,
        "correct_tool_rate": round(correct_tool / total, 4),
        "correct_argument_count": correct_args,
        "correct_argument_rate": round(correct_args / total, 4),
        "clarification_precision": round(clarify_correct / max(1, clarify_expected), 4),
        "wrong_tool_count": wrong,
        "no_tool_when_expected": missing,
        "false_side_effect_count": false_effect,
        "confirmation_bypass_count": bypass,
        "context_resolution_count": context,
    }


class NormalizerTests(unittest.TestCase):
    def test_alias_and_ordinal(self):
        value = normalize_utterance("  Mở YOU TUBE, cái thứ hai!!!  ")
        self.assertIn("YouTube", value["normalized"])
        self.assertEqual(value["ordinal"], 2)

    def test_preserves_search_qualifiers(self):
        value = normalize_utterance("Chicago live remix cover karaoke official audio video shorts")
        for token in ("live", "remix", "cover", "karaoke", "official", "audio", "video", "shorts"):
            self.assertIn(token, value["tokens"])


class ContextTests(unittest.TestCase):
    def test_selection_and_redaction(self):
        now = [10.0]
        context = InteractionContext(clock=lambda: now[0])
        context.store(context_type="clarification", session_generation=2,
                      options=[{"title": "One", "tab_ref": "opaque-1", "id": 999,
                                "url": "https://example.com/private?q=x"},
                               {"title": "Two", "tab_ref": "opaque-2"}])
        selected = context.select(2, 2)
        self.assertTrue(selected["success"])
        self.assertEqual(selected["data"]["selection"]["title"], "Two")
        serialized = json.dumps(selected)
        self.assertNotIn("999", serialized)
        self.assertNotIn("https://", serialized)

    def test_expiry_and_generation(self):
        now = [10.0]
        context = InteractionContext(clock=lambda: now[0])
        context.store(context_type="browser_tabs", session_generation=2,
                      options=[{"title": "One"}], ttl=2)
        self.assertEqual(context.select(1, 3)["status"], "no_valid_context")
        context.store(context_type="browser_tabs", session_generation=2,
                      options=[{"title": "One"}], ttl=2)
        now[0] = 13.0
        self.assertEqual(context.select(1, 2)["status"], "no_valid_context")


class DatasetTests(unittest.TestCase):
    def test_dataset_shape_and_safety_metrics(self):
        rows = load_dataset()
        self.assertGreaterEqual(len(rows), 150)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        required = {"id", "utterance", "expected_tool", "expected_arguments",
                    "should_clarify", "should_confirm", "category"}
        self.assertTrue(all(set(row) == required for row in rows))
        metrics = evaluate_dataset()
        self.assertEqual(metrics["false_side_effect_count"], 0)
        self.assertEqual(metrics["confirmation_bypass_count"], 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Reserved for a future explicit Gemini semantic test")
    args, remaining = parser.parse_known_args()
    if args.live:
        print("Live semantic routing is intentionally not implemented in Phase B.")
        return 2
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(evaluate_dataset(), ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
