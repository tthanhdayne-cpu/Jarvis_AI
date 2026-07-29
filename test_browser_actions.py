"""Run one browser action without Gemini, OpenWakeWord, Jarvis UI, or browser launch."""

from __future__ import annotations

import argparse
import json
import threading
import time

from actions.windows_action_registry import ActionRuntime, WINDOWS_ACTION_REGISTRY
from actions.windows_browser_actions import inspect_foreground_window


ACTION_MAP = {
    "new_tab": "browser_new_tab",
    "next_tab": "browser_next_tab",
    "previous_tab": "browser_previous_tab",
    "focus_address_bar": "browser_focus_address_bar",
    "search": "browser_search",
    "type_text": "browser_type_text",
    "press_enter": "browser_press_enter",
    "scroll": "browser_scroll",
    "find_text": "browser_find_text",
}
BLOCKED_TESTS = {
    "blocked_notepad": ("browser_new_tab", {}, "browser_not_foreground"),
    "blocked_otp": ("browser_type_text", {"text": "123456"}, "sensitive_input_blocked"),
    "blocked_api_key": (
        "browser_type_text",
        {"text": "api_key=sk-example-secret-value"},
        "sensitive_input_blocked",
    ),
    "blocked_empty_query": ("browser_search", {"query": ""}, "validation_failed"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run exactly one SAFE browser action")
    parser.add_argument("action", choices=tuple(ACTION_MAP) + tuple(BLOCKED_TESTS))
    parser.add_argument("--query")
    parser.add_argument("--text")
    parser.add_argument("--direction", choices=("up", "down"))
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Seconds to focus the target window before the single action runs",
    )
    return parser.parse_args()


def resolve_action(args):
    if args.action in BLOCKED_TESTS:
        action_name, parameters, expected_status = BLOCKED_TESTS[args.action]
        return action_name, parameters, expected_status
    action_name = ACTION_MAP[args.action]
    if args.action == "search":
        if args.query is None:
            raise SystemExit("--query is required for search")
        return action_name, {"query": args.query}, None
    if args.action in {"type_text", "find_text"}:
        if args.text is None:
            raise SystemExit(f"--text is required for {args.action}")
        return action_name, {"text": args.text}, None
    if args.action == "scroll":
        if args.direction is None:
            raise SystemExit("--direction up/down is required for scroll")
        return action_name, {"direction": args.direction}, None
    return action_name, {}, None


def main():
    args = parse_args()
    action_name, parameters, expected_status = resolve_action(args)
    if args.action == "blocked_notepad":
        print("Preflight: focus a real Notepad window before running this test.")
    elif args.action != "blocked_empty_query":
        print("Preflight: focus the real browser/profile you want to test.")

    if args.action != "blocked_empty_query" and args.delay > 0:
        print(f"Waiting {args.delay:.1f}s for you to focus the target window...")
        time.sleep(min(args.delay, 15.0))

    foreground = inspect_foreground_window()
    print(json.dumps({
        "foreground_hwnd": foreground.hwnd,
        "foreground_pid": foreground.pid,
        "foreground_process": foreground.process,
        "foreground_is_browser": foreground.is_browser,
    }, ensure_ascii=False, indent=2))

    generation = 1
    runtime = ActionRuntime(
        state_getter=lambda: "ACTIVE",
        sleep_intent_getter=lambda: False,
        generation_getter=lambda: generation,
        expected_generation=generation,
        cancellation_event=threading.Event(),
    )
    try:
        result = WINDOWS_ACTION_REGISTRY.dispatch(
            action_name, parameters, runtime
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if expected_status:
            assert result.get("status") == expected_status, (
                f"Expected {expected_status}, got {result.get('status')}"
            )
    finally:
        WINDOWS_ACTION_REGISTRY.shutdown(wait=True)


if __name__ == "__main__":
    main()
