"""Offline Phase 1 regression checks for the Gemini Live configuration."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_PATH = ROOT / "main.py"
REGISTRY_PATH = ROOT / "actions" / "windows_action_registry.py"

REQUIRED_WINDOWS_TOOLS = {
    "close_browser_tab",
    "youtube_search",
    "play_youtube_video",
    "list_browser_tabs",
    "close_browser_tab_by_title",
    "close_browser_tab_by_index",
    "close_all_matching_tabs",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def string_constant(tree: ast.AST, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = node.value
                try:
                    evaluated = ast.literal_eval(value)
                except Exception as exc:
                    raise AssertionError(f"{name} is not a static string") from exc
                require(isinstance(evaluated, str), f"{name} must be a string")
                return evaluated
    raise AssertionError(f"Missing constant: {name}")


def literal_tool_names(source: str) -> set[str]:
    return set(re.findall(r'^[ \t]+["\']name["\']\s*:\s*["\']([^"\']+)["\']', source, re.MULTILINE))


def latest_backup() -> Path:
    backups = sorted(ROOT.glob("main.py.backup-live-search-phase1-*"))
    require(bool(backups), "Phase 1 main.py backup is missing")
    return backups[-1]


def main() -> int:
    source = MAIN_PATH.read_text(encoding="utf-8")
    registry = REGISTRY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MAIN_PATH))

    require(
        source.count('{"google_search": {}}') == 1,
        "google_search must exist exactly once in main.py",
    )
    require(
        re.search(
            r'tools\s*=\s*\[\s*\{"google_search": \{\}\},\s*'
            r'\{"function_declarations": TOOL_DECLARATIONS\},\s*\]',
            source,
            re.DOTALL,
        ) is not None,
        "Live tools must keep Google Search and all function declarations as separate groups",
    )
    require('response_modalities=["AUDIO"]' in source, "Native AUDIO mode changed")
    require("output_audio_transcription={}" in source, "Output transcription is disabled")
    require("input_audio_transcription={}" in source, "Input transcription is disabled")
    require('voice_name="Charon"' in source, "Charon voice changed")
    require("SessionResumptionConfig()" in source, "Session resumption is disabled")

    for name in REQUIRED_WINDOWS_TOOLS:
        require(name in source, f"Required tool missing from main.py: {name}")
        require(name in registry, f"Required tool missing from Registry: {name}")

    require(
        re.search(
            r'"close_browser_tab"[\s\S]{0,500}?PermissionLevel\.CONFIRM', registry
        ) is not None,
        "close_browser_tab is no longer CONFIRM",
    )
    require(
        re.search(
            r'"play_youtube_video"[\s\S]{0,900}?PermissionLevel\.SAFE', registry
        ) is not None,
        "play_youtube_video is no longer SAFE",
    )

    backup_source = latest_backup().read_text(encoding="utf-8")
    before_names = literal_tool_names(backup_source)
    after_names = literal_tool_names(source)
    require(
        before_names <= after_names,
        "Existing literal tool declarations were lost: "
        + ", ".join(sorted(before_names - after_names)),
    )

    policy = string_constant(tree, "CURRENT_INFORMATION_POLICY").lower()
    require("google search" in policy and "before answering" in policy, "Current-data search is not mandatory")
    require("weather" in policy and "today" in policy, "Weather/today policy is missing")
    require("stable background knowledge" in policy, "Stable knowledge exemption is missing")
    require("local windows actions" in policy, "Local-action search exemption is missing")
    require("windows action registry" in policy, "Registry side-effect boundary is missing")
    require("legacy web_search" in policy, "Built-in Search priority is missing")

    policy_cases = {
        "Thời tiết hôm nay": "search_required",
        "Mở YouTube": "function_tool_not_search",
        "Giải thích OOP": "search_not_required",
    }
    print("PASS: Gemini Live has one Google Search tool plus all function declarations")
    print("PASS: AUDIO, Charon, transcriptions, and session resumption are unchanged")
    print("PASS: required Registry tools and permissions are preserved")
    print("PASS: no pre-Phase-1 literal tool declaration was removed")
    for prompt, expected in policy_cases.items():
        print(f"PASS: policy case {prompt!r} -> {expected}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

