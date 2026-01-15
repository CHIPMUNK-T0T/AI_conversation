import argparse
import os
import sys
import tempfile

from faster_whisper import WhisperModel
from openai import OpenAI
import sounddevice as sd
import soundfile as sf


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
        "--device",
        default="cuda",
        help="Device for inference, e.g. cpu/cuda (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help=(
            "CTranslate2 compute type (default: float16/fp16). "
            "Common values: float16, float32, int8, int8_float16, int8_bfloat16, auto."
        ),
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=5.0,
        help="Recording length in seconds when using the mic (default: 5)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for mic recording (default: 16000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = args.audio_path
    temp_path = None
    if audio_path:
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}", file=sys.stderr)
            return 1
    else:
        print("音声取得中...")
        print(f"Recording {args.record_seconds:.1f}s from microphone...")
        recording = sd.rec(
            int(args.sample_rate * args.record_seconds),
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, recording, args.sample_rate)
        audio_path = temp_path

    model = WhisperModel("large-v3-turbo", device=args.device, compute_type=args.compute_type)
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
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
    )
    res = client.chat.completions.create(
        model="llama-3.1-swallow-8b-instruct-v0.5",
        messages=[{"role": "user", "content": transcript}],
    )
    print("出力:")
    print(res.choices[0].message.content)
    if temp_path:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
