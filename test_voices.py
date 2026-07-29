import asyncio
import json
import sys
from pathlib import Path

import sounddevice as sd
from google import genai
from google.genai import types


MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
SAMPLE_RATE = 24000
VOICE_TIMEOUT_SECONDS = 30

VIETNAMESE_SAMPLE = "Xin chào, tôi là Jarvis. Tôi đã sẵn sàng hỗ trợ bạn."
ENGLISH_SAMPLE = "Hello, I am Jarvis. I am ready to assist you."
VIETNAMESE_AUDITION_SAMPLE = (
    "Xin chào, tôi là Jarvis. Hôm nay là thứ Bảy, ngày 11 tháng 7. "
    "Tôi có thể giúp bạn mở ứng dụng, tìm kiếm thông tin và quản lý máy tính. "
    "Bạn muốn tôi hỗ trợ việc gì?"
)

QUICK_VOICES = [
    "Charon", "Fenrir", "Orus", "Kore",
    "Puck", "Zephyr", "Aoede", "Leda",
]

ALL_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
    "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus",
    "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome",
    "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam",
    "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

BASE_DIR = Path(__file__).resolve().parent
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"


def load_api_key() -> str:
    with API_CONFIG_PATH.open("r", encoding="utf-8") as file:
        key = json.load(file).get("gemini_api_key", "")
    if not key:
        raise RuntimeError("Không tìm thấy Gemini API key trong cấu hình.")
    return key


def choose_mode() -> tuple[list[str], bool]:
    print("Chọn chế độ:")
    print("  1: thử nhanh tiếng Việt")
    print("  2: thử toàn bộ 30 voice bằng tiếng Việt")
    print("  3: thử cả Việt và Anh như hiện tại")
    while True:
        choice = input("Lựa chọn [1/2/3]: ").strip()
        if choice == "1":
            return QUICK_VOICES, True
        if choice == "2":
            return ALL_VOICES, True
        if choice == "3":
            while True:
                scope = input("Thử nhanh 8 voice (1) hay toàn bộ 30 voice (2)? ").strip()
                if scope == "1":
                    return QUICK_VOICES, False
                if scope == "2":
                    return ALL_VOICES, False
                print("Vui lòng nhập 1 hoặc 2.")
        print("Vui lòng nhập 1, 2 hoặc 3.")


def make_live_config(voice_name: str) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=(
            "You are a voice audition system. Read the supplied sample sentence "
            "exactly as written. Do not add, remove, translate, explain, or greet."
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
    )


async def play_sentence(session, stream, text: str) -> None:
    await session.send_client_content(
        turns={"parts": [{"text": f'Read exactly: "{text}"'}]},
        turn_complete=True,
    )

    async with asyncio.timeout(VOICE_TIMEOUT_SECONDS):
        async for response in session.receive():
            if response.data:
                await asyncio.to_thread(stream.write, response.data)
            if response.server_content and response.server_content.turn_complete:
                return

    raise TimeoutError("Gemini không báo turn_complete.")


async def audition_voice(
    client: genai.Client,
    voice_name: str,
    number: int,
    total: int,
    vietnamese_only: bool,
) -> bool:
    stream = None
    try:
        stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            latency="low",
        )
        stream.start()

        config = make_live_config(voice_name)
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print(f"\n[{number}/{total}] Voice: {voice_name} | Language: Vietnamese")
            sample = VIETNAMESE_AUDITION_SAMPLE if vietnamese_only else VIETNAMESE_SAMPLE
            await play_sentence(session, stream, sample)

            if not vietnamese_only:
                print(f"[{number}/{total}] Voice: {voice_name} | Language: English")
                await play_sentence(session, stream, ENGLISH_SAMPLE)

        return True
    except TimeoutError:
        print(f"[ERROR] {voice_name}: timeout sau {VOICE_TIMEOUT_SECONDS} giây.")
    except Exception as error:
        message = " ".join(str(error).split())[:180]
        print(f"[ERROR] {voice_name}: {message or type(error).__name__}")
    finally:
        if stream is not None:
            try:
                if stream.active:
                    stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
    return False


async def async_main() -> int:
    voices, vietnamese_only = choose_mode()
    try:
        api_key = load_api_key()
    except Exception as error:
        print(f"[ERROR] {error}")
        return 1

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta"},
    )
    vietnamese_voice = None
    english_voice = None
    ratings: dict[str, int] = {}

    index = 0
    while index < len(voices):
        voice_name = voices[index]
        played = await audition_voice(
            client, voice_name, index + 1, len(voices), vietnamese_only
        )
        if not played:
            index += 1
            continue

        while True:
            if vietnamese_only:
                prompt = (
                    "Enter=tiếp | r=nghe lại | 1-5=chấm điểm | "
                    "v=chọn giọng Việt | q=kết thúc: "
                )
            else:
                prompt = (
                    "Enter=tiếp | v=giọng Việt | e=giọng Anh | "
                    "r=nghe lại | s=bỏ qua | q=thoát: "
                )
            command = (await asyncio.to_thread(input, prompt)).strip().casefold()

            if vietnamese_only and command in {"1", "2", "3", "4", "5"}:
                ratings[voice_name] = int(command)
                print(f"Đã chấm {voice_name}: {command}/5")
            elif command == "v":
                vietnamese_voice = voice_name
                print(f"Đã chọn giọng Việt: {voice_name}")
            elif not vietnamese_only and command == "e":
                english_voice = voice_name
                print(f"Đã chọn giọng Anh: {voice_name}")
            elif command == "r":
                break
            elif command == "" or (not vietnamese_only and command == "s"):
                index += 1
                break
            elif command == "q":
                index = len(voices)
                break
            else:
                print("Phím không hợp lệ.")

        if command == "r":
            continue

    print("\nKết quả lựa chọn:")
    if vietnamese_only:
        ranked = sorted(ratings.items(), key=lambda item: (-item[1], voices.index(item[0])))
        print("Top 5 voice tiếng Việt:")
        if ranked:
            for rank, (voice_name, score) in enumerate(ranked[:5], start=1):
                print(f"  {rank}. {voice_name}: {score}/5")
        else:
            print("  Chưa có voice nào được chấm điểm.")
    print(f"Vietnamese voice: {vietnamese_voice or 'Not selected'}")
    if not vietnamese_only:
        print(f"English voice: {english_voice or 'Not selected'}")
    return 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nĐã dừng thử giọng.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
