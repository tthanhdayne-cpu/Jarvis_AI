"""Phase 0 compatibility probe for Gemini Live search and function tools.

This script is intentionally independent from main.py. It sends text prompts,
receives native-audio bytes without playing them, and never invokes a local
action. Run it only while the main Jarvis process is stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
MAIN_PATH = ROOT / "main.py"
KEY_PATH = ROOT / "config" / "api_keys.json"
DEFAULT_TIMEOUT_SECONDS = 45.0

ECHO_DECLARATION = {
    "name": "compatibility_echo",
    "description": "Compatibility-only function. No side effect.",
    "parameters": {
        "type": "OBJECT",
        "properties": {"text": {"type": "STRING"}},
        "required": ["text"],
    },
}


@dataclass
class TurnObservation:
    response_received: bool = False
    audio_received: bool = False
    audio_chunks: int = 0
    audio_bytes: int = 0
    output_transcription_received: bool = False
    input_transcription_received: bool = False
    turn_complete: bool = False
    tool_called: bool = False
    tool_response_accepted: bool = False
    grounding_metadata_present: bool = False
    grounding_chunks_count: int = 0
    grounding_supports_count: int = 0
    search_entry_point_present: bool = False
    source_domains: list[str] = field(default_factory=list)


def read_configured_model() -> str:
    """Read LIVE_MODEL as source text without importing main.py."""
    source = MAIN_PATH.read_text(encoding="utf-8")
    match = re.search(r'^LIVE_MODEL\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    if not match:
        raise RuntimeError("LIVE_MODEL was not found in main.py")
    return match.group(1)


def load_api_key() -> str:
    """Load the key only at execution time; never print or return it in results."""
    import os

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        data = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("Gemini credential file is missing") from exc
    key = str(data.get("gemini_api_key", "")).strip()
    if not key:
        raise RuntimeError("Gemini API key is not configured")
    return key


def safe_error(exc: BaseException) -> str:
    """Return a bounded diagnostic with obvious credential-like text removed."""
    message = re.sub(
        r"(?:AIza|AQ\.)[A-Za-z0-9._-]{12,}", "[REDACTED]", str(exc)
    )
    message = re.sub(r"https?://\S+", "[URL_REDACTED]", message)
    return f"{type(exc).__name__}: {message[:500]}"


def classify_connection_error(message: str) -> str | None:
    lowered = message.lower()
    if any(token in lowered for token in ("not found", "404", "unknown model")):
        return "model_not_found"
    if any(token in lowered for token in ("deprecated", "no longer available")):
        return "model_deprecated"
    return None


def _domain(uri: str) -> str | None:
    try:
        hostname = urlsplit(uri).hostname
        return hostname.lower() if hostname else None
    except Exception:
        return None


def observe_grounding(metadata: Any, observation: TurnObservation) -> None:
    if metadata is None:
        return
    observation.grounding_metadata_present = True
    chunks = list(getattr(metadata, "grounding_chunks", None) or [])
    supports = list(getattr(metadata, "grounding_supports", None) or [])
    observation.grounding_chunks_count += len(chunks)
    observation.grounding_supports_count += len(supports)
    observation.search_entry_point_present |= bool(
        getattr(metadata, "search_entry_point", None)
    )
    domains = set(observation.source_domains)
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        domain = _domain(str(getattr(web, "uri", ""))) if web else None
        if domain:
            domains.add(domain)
    observation.source_domains = sorted(domains)


class AudioSink:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._wave: wave.Wave_write | None = None

    def write(self, data: bytes) -> None:
        if not data or self.path is None:
            return
        if self._wave is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._wave = wave.open(str(self.path), "wb")
            self._wave.setnchannels(1)
            self._wave.setsampwidth(2)
            self._wave.setframerate(24000)
        self._wave.writeframesraw(data)

    def close(self) -> None:
        if self._wave is not None:
            self._wave.close()
            self._wave = None


def build_config(types: Any, mode: str) -> Any:
    tools: list[dict[str, Any]] = []
    if mode in {"search", "combined"}:
        tools.append({"google_search": {}})
    if mode in {"function", "combined"}:
        tools.append({"function_declarations": [ECHO_DECLARATION]})
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription={},
        output_audio_transcription={},
        tools=tools,
    )


async def collect_turn(
    session: Any,
    types: Any,
    prompt: str,
    timeout: float,
    audio_sink: AudioSink,
    expect_tool: bool = False,
) -> TurnObservation:
    observation = TurnObservation()
    await session.send_client_content(
        turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
        turn_complete=True,
    )

    async with asyncio.timeout(timeout):
        async for message in session.receive():
            content = getattr(message, "server_content", None)
            if content is not None:
                observation.response_received = True
                observation.turn_complete |= bool(getattr(content, "turn_complete", False))
                observation.input_transcription_received |= bool(
                    getattr(content, "input_transcription", None)
                )
                observation.output_transcription_received |= bool(
                    getattr(content, "output_transcription", None)
                )
                observe_grounding(getattr(content, "grounding_metadata", None), observation)

            data = getattr(message, "data", None)
            if data:
                observation.audio_received = True
                observation.audio_chunks += 1
                observation.audio_bytes += len(data)
                audio_sink.write(data)

            tool_call = getattr(message, "tool_call", None)
            calls = list(getattr(tool_call, "function_calls", None) or [])
            for call in calls:
                if getattr(call, "name", None) != "compatibility_echo":
                    continue
                observation.tool_called = True
                args = dict(getattr(call, "args", None) or {})
                response = types.FunctionResponse(
                    id=call.id,
                    name="compatibility_echo",
                    response={
                        "ok": True,
                        "echo": str(args.get("text", ""))[:200],
                        "side_effect": False,
                    },
                )
                await session.send_tool_response(function_responses=response)
                observation.tool_response_accepted = True

            if observation.turn_complete and (not expect_tool or observation.tool_response_accepted):
                break
    return observation


def mode_result(connected: bool, observation: TurnObservation | None, error: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"connected": connected}
    if observation is not None:
        result.update(asdict(observation))
    if error:
        result["error"] = error
        category = classify_connection_error(error)
        if category:
            result["error_category"] = category
    return result


async def run_single_mode(
    mode: str,
    model: str,
    timeout: float,
    audio_path: Path | None,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    connected = False
    sink = AudioSink(audio_path)
    try:
        client = genai.Client(api_key=load_api_key())
        config = build_config(types, mode)
        async with asyncio.timeout(timeout):
            async with client.aio.live.connect(model=model, config=config) as session:
                connected = True
                if mode == "function":
                    obs = await collect_turn(
                        session,
                        types,
                        "Call compatibility_echo with text phase-zero. Do not answer before calling it.",
                        timeout,
                        sink,
                        expect_tool=True,
                    )
                    return mode_result(True, obs, None)

                if mode == "search":
                    obs = await collect_turn(
                        session,
                        types,
                        "Hôm nay là ngày bao nhiêu và tài liệu Gemini API gần đây nhất được cập nhật khi nào?",
                        timeout,
                        sink,
                    )
                    return mode_result(True, obs, None)

                function_obs = await collect_turn(
                    session,
                    types,
                    "Call compatibility_echo with text combined-phase-zero. Do not answer before calling it.",
                    timeout,
                    sink,
                    expect_tool=True,
                )
                search_obs = await collect_turn(
                    session,
                    types,
                    "Hôm nay là ngày bao nhiêu và tài liệu Gemini API gần đây nhất được cập nhật khi nào?",
                    timeout,
                    sink,
                )
                combined = TurnObservation(
                    response_received=search_obs.response_received,
                    audio_received=function_obs.audio_received or search_obs.audio_received,
                    audio_chunks=function_obs.audio_chunks + search_obs.audio_chunks,
                    audio_bytes=function_obs.audio_bytes + search_obs.audio_bytes,
                    output_transcription_received=(
                        function_obs.output_transcription_received
                        or search_obs.output_transcription_received
                    ),
                    input_transcription_received=(
                        function_obs.input_transcription_received
                        or search_obs.input_transcription_received
                    ),
                    turn_complete=function_obs.turn_complete and search_obs.turn_complete,
                    tool_called=function_obs.tool_called,
                    tool_response_accepted=function_obs.tool_response_accepted,
                    grounding_metadata_present=search_obs.grounding_metadata_present,
                    grounding_chunks_count=search_obs.grounding_chunks_count,
                    grounding_supports_count=search_obs.grounding_supports_count,
                    search_entry_point_present=search_obs.search_entry_point_present,
                    source_domains=search_obs.source_domains,
                )
                result = mode_result(True, combined, None)
                result["function_worked"] = bool(
                    function_obs.tool_called and function_obs.tool_response_accepted
                )
                result["search_worked"] = bool(
                    search_obs.response_received and search_obs.turn_complete
                )
                result["session_remained_connected"] = True
                return result
    except asyncio.TimeoutError:
        return mode_result(connected, None, f"TimeoutError: exceeded {timeout:g} seconds")
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        return mode_result(connected, None, safe_error(exc))
    finally:
        sink.close()


def audio_path_for(base: Path | None, mode: str, multiple: bool) -> Path | None:
    if base is None:
        return None
    if not multiple:
        return base
    suffix = base.suffix or ".wav"
    return base.with_name(f"{base.stem}-{mode}{suffix}")


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    configured_model = read_configured_model()
    model = args.model or configured_model
    modes = ["function", "search", "combined"] if args.mode == "all" else [args.mode]
    results: dict[str, Any] = {
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "configured_model": configured_model,
        "model_overridden": bool(args.model),
        "audio_playback": False,
    }
    for mode in modes:
        results[f"{mode}_only" if mode != "combined" else "combined"] = await run_single_mode(
            mode,
            model,
            args.timeout,
            audio_path_for(args.save_audio, mode, len(modes) > 1),
        )

    combined = results.get("combined", {})
    results["compatible"] = bool(
        combined.get("connected")
        and combined.get("function_worked")
        and combined.get("search_worked")
        and combined.get("audio_received")
        and combined.get("tool_response_accepted")
        and combined.get("session_remained_connected")
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("function", "search", "combined", "all"), default="all"
    )
    parser.add_argument("--model", help="Manually override the model for this probe only")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--save-audio",
        nargs="?",
        const=Path("compatibility-audio.wav"),
        type=Path,
        help="Optionally save received 24 kHz mono PCM as WAV; audio is never played",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        result = {
            "model": args.model or "configured model (not loaded)",
            "compatible": False,
            "cancelled": True,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("compatible") else 1


if __name__ == "__main__":
    sys.exit(main())

