# Current Information Compatibility — Phase 0

This phase is an isolated compatibility probe. It does not modify or import
`main.py`, does not invoke Windows actions, does not use the microphone, and
does not play received audio. Stop Jarvis before running the probe so a second
Live session is not created beside the active assistant.

The default model is read from `LIVE_MODEL` in `main.py`. At the time this probe
was created it resolves to:

`models/gemini-2.5-flash-native-audio-preview-12-2025`

The probe exercises three configurations:

1. Function only: a side-effect-free `compatibility_echo` declaration and a
   manually submitted `FunctionResponse`.
2. Search only: the built-in `google_search` tool and a harmless current-data
   question.
3. Combined: both tool groups in one native-audio session, with a function turn
   followed by a search turn.

Audio is counted in memory and discarded by default. `--save-audio` writes a
24 kHz, mono, 16-bit PCM WAV file without playback. The JSON report includes
only redacted source domains, metadata counts, and bounded error details. It
never prints the API key, cookies, full response payload, or source URLs.

## Commands

```bat
cd /d D:\JARVIS\Mark-XXXIX-OR

venv\Scripts\python.exe test_live_search_compatibility.py --mode function
venv\Scripts\python.exe test_live_search_compatibility.py --mode search
venv\Scripts\python.exe test_live_search_compatibility.py --mode combined
venv\Scripts\python.exe test_live_search_compatibility.py --mode all
```

Optional arguments:

```text
--model <model-name>
--timeout <seconds>
--save-audio [output.wav]
```

`compatible` is true only when the combined configuration connects, the echo
function is called, its manual response is accepted, the subsequent search turn
finishes, native audio is received, and the session remains connected. Search
grounding metadata is reported when Live supplies it; its absence is recorded
as an API limitation and is not treated as proof that citations exist.

Phase 1 must not begin until the combined probe returns `compatible: true` for
the intended production model. A model-not-found, deprecated-model, tool schema
conflict, missing audio, failed manual function response, or disconnected
combined session blocks Phase 1.

