"""Run exactly one SAFE Windows action without Jarvis, Gemini, wake word, or UI."""

from __future__ import annotations

import argparse
import json
import threading

from actions.windows_action_registry import ActionRuntime, WINDOWS_ACTION_REGISTRY


TEST_ACTIONS = (
    "get_system_status",
    "list_processes",
    "take_screenshot",
    "set_volume",
    "focus_window",
    "minimize_window",
    "maximize_window",
    "restore_window",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one SAFE Windows action in isolation."
    )
    parser.add_argument("action", choices=TEST_ACTIONS)
    parser.add_argument("--name", help="Window process name or title")
    parser.add_argument("--level", type=int, help="Volume level from 0 to 100")
    parser.add_argument("--limit", type=int, default=20, help="Process result limit")
    return parser.parse_args()


def parameters_for(args) -> dict:
    if args.action in {
        "focus_window", "minimize_window", "maximize_window", "restore_window"
    }:
        if not args.name:
            raise SystemExit("--name is required for window actions")
        return {"name": args.name}
    if args.action == "set_volume":
        if args.level is None:
            raise SystemExit("--level is required for set_volume")
        return {"level": args.level}
    if args.action == "list_processes":
        return {"limit": args.limit}
    return {}


def normalize_process(value: str) -> str:
    normalized = str(value or "").casefold().strip()
    if normalized.endswith(".exe"):
        normalized = normalized[:-4]
    return normalized.replace("_", " ").replace("-", " ").strip()


def assert_window_match(args, result: dict) -> None:
    if args.action not in {
        "focus_window", "minimize_window", "maximize_window", "restore_window"
    } or not result.get("success"):
        return
    process = normalize_process(result.get("data", {}).get("process", ""))
    terminal_processes = {"windowsterminal", "cmd", "powershell", "pwsh"}
    requested = normalize_process(args.name)
    if requested == "notepad":
        assert process == "notepad", (
            f"Expected notepad process, got {process or '<empty>'}"
        )
    if requested not in {"terminal", "cmd", "powershell", "pwsh"}:
        assert process not in terminal_processes, (
            f"Terminal process was incorrectly selected: {process}"
        )


def main():
    args = parse_args()
    generation = 1
    cancel_event = threading.Event()
    runtime = ActionRuntime(
        state_getter=lambda: "ACTIVE",
        sleep_intent_getter=lambda: False,
        generation_getter=lambda: generation,
        expected_generation=generation,
        cancellation_event=cancel_event,
    )
    try:
        if args.action in {
            "focus_window", "minimize_window", "maximize_window", "restore_window"
        }:
            print(
                f"Preflight: open the real '{args.name}' application/window before continuing."
            )
        result = WINDOWS_ACTION_REGISTRY.dispatch(
            args.action, parameters_for(args), runtime
        )
        assert_window_match(args, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        WINDOWS_ACTION_REGISTRY.shutdown(wait=True)


if __name__ == "__main__":
    main()
