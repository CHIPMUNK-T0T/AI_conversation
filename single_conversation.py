import argparse
import io
import os
import sys
import tempfile

import requests
from faster_whisper import WhisperModel
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe Japanese audio to text using faster-whisper."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Path to an audio file (wav/mp3/etc.). If omitted, record from mic.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=5.0,
        help="Recording length in seconds when using the mic (default: 5)",
    )
    parser.add_argument(
        "--voicevox-speaker",
        type=int,
        default=3,
        help="VOICEVOX speaker ID (default: 1)",
    )
    return parser.parse_args()


def synthesize_voice(text: str, base_url: str, speaker: int) -> bytes:
    query_res = requests.post(
        f"{base_url}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=60,
    )
    query_res.raise_for_status()
    synthesis_res = requests.post(
        f"{base_url}/synthesis",
        params={"speaker": speaker},
        json=query_res.json(),
        timeout=120,
    )
    synthesis_res.raise_for_status()
    return synthesis_res.content


def main() -> int:
    args = parse_args()
    audio_path = args.audio_path
    temp_path = None
    if audio_path:
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}", file=sys.stderr)
            return 1
    else:
        import sounddevice as sd
        import soundfile as sf

        print("音声取得中...")
        print(f"Recording {args.record_seconds:.1f}s from microphone...")
        recording = sd.rec(
            int(16000 * args.record_seconds),
            samplerate=16000,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, recording, 16000)
        audio_path = temp_path

    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print("テキスト出力中...")
    segments, _info = model.transcribe(
        audio_path,
        language="ja",
    )

    texts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            texts.append(text)
    transcript = " ".join(texts)
    print(transcript)

    print("LLMに依頼中...")
    client = OpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    res = client.chat.completions.create(
        model=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8"),
        messages=[{"role": "user", "content": transcript}],
    )
    print("出力:")
    answer_text = res.choices[0].message.content
    print(answer_text)
    print("音声合成中...")
    wav_bytes = synthesize_voice(
        answer_text,
        base_url="http://127.0.0.1:50021",
        speaker=args.voicevox_speaker,
    )
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        sd.play(data, sr)
        sd.wait()
    except OSError as exc:
        print(f"再生に失敗しました: {exc}", file=sys.stderr)
    if temp_path:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
