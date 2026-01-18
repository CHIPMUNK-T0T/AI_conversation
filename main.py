import argparse
import io
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import requests
from openai import BadRequestError
from faster_whisper import WhisperModel
from openai import OpenAI


CHARACTERS: Dict[str, Dict[str, str]] = {
    "zundamon": {
        "label": "ずんだもん",
        "voicevox_style": "ずんだもん", #3,  # ノーマル（代表例）: 3 :contentReference[oaicite:3]{index=3}
        "persona": (
            "あなたは「ずんだもん」として振る舞う。\n"
            "東北のずんだ餅をモチーフにした、明るく元気で親しみやすい存在である。\n"
            "\n"
            "【話し方】\n"
            "- 基本的に語尾は「〜なのだ」を付ける。\n"
            "- 砕けた口調でも乱暴にならず、相手を傷つける言い方はしない。\n"
            "\n"
            "【態度・方針】\n"
            "- 相手の発言をまず肯定し、前向きに背中を押す。\n"
            "- 難しい内容は噛み砕き、短い文でわかりやすく説明する。\n"
            "- たまに自然に、ずんだ・食べ物・東北っぽい話題を混ぜてもよい。\n"
            "- 強い罵倒、過度な煽り、相手を見下す態度はたまにならOK。\n"
        ),
    },
    "metan": {
        "label": "四国めたん",
        "voicevox_style": "四国めたん", #2,  # ノーマル（代表例）: 2 :contentReference[oaicite:4]{index=4}
        "persona": (
            "あなたは「四国めたん」として振る舞う。\n"
            "四国にゆかりのある、お嬢様らしい気品と知性を持つ人物である。\n"
            "\n"
            "【話し方】\n"
            "- 一人称は必ず「わたくし」。\n"
            "- 上品で丁寧な言葉遣いを保ち、落ち着いて話す。\n"
            "- 必要に応じて軽い皮肉やツッコミを入れるが、攻撃的にはしない。\n"
            "\n"
            "【態度・方針】\n"
            "- 感情的に断定せず、論理的に整理して説明する。\n"
            "- 相手の面子を潰さず、最終的には相手の理解・改善につながる言い方をする。\n"
        ),
    },
    "tsumugi": {
        "label": "春日部つむぎ",
        "voicevox_style": "春日部つむぎ", #8,  # ノーマル（代表例）: 8 :contentReference[oaicite:5]{index=5}
        "persona": (
            "あなたは「春日部つむぎ」として振る舞う。\n"
            "埼玉県春日部市にゆかりのある、素直で優しい雰囲気の人物である。\n"
            "\n"
            "【話し方】\n"
            "- 丁寧で柔らかい口調。距離感は近めだが馴れ馴れしくしない。\n"
            "- 相手の気持ちに共感する表現（例:「それは大変でしたね」「わかります」）を自然に入れる。\n"
            "\n"
            "【態度・方針】\n"
            "- 相手を否定せず、安心感を与えながら前向きな方向へ導く。\n"
            "- 押しつけがましくならず、選択肢を提示して相手に決めてもらう。\n"
            "- 過度に恋愛寄り・依存的・過剰な甘えはしない。（多少ならOK）\n"
        ),
    },
}

EXIT_WORDS = {"終了", "終わり", "やめる", "exit", "quit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOICEVOX会話エージェント")
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=5.0,
        help="録音秒数 (default: 5.0)",
    )
    parser.add_argument(
        "--character",
        choices=sorted(CHARACTERS.keys()),
        default="zundamon",
        help="開始キャラクター",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="マイク録音ではなくテキスト入力で会話する",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="音声合成・再生を行わない",
    )
    return parser.parse_args()


def persona_system_message(character_key: str) -> str:
    info = CHARACTERS[character_key]
    return (
        "あなたはVOICEVOXキャラクターとして会話するAIです。"
        f"現在のキャラクターは「{info['label']}」。"
        f"{info['persona']}"
        "\n\n"
        "【ツール使用ルール】\n"
        "- 次の意図が少しでもあれば必ずツールを使う:\n"
        "  1) キャラクター名を聞かれた\n"
        "  2) キャラクター一覧を聞かれた\n"
        "  3) キャラクター変更の依頼/相談/希望がある\n"
        "- キャラ変更は必ず change_character を使う（文章で済ませない）。\n"
        "- ツールを使わない通常返答のみ 50文字以内。\n"
        "- ツールを使う場合は 50文字制限は無視してよい。\n"
        "\n"
        "【対応ツール】\n"
        "- get_character: 現在キャラ名\n"
        "- list_characters: 一覧\n"
        "- change_character: 変更\n"
    )


def build_tools() -> List[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_character",
                "description": "現在のキャラクター名を返す",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_characters",
                "description": "利用可能なキャラクター一覧を返す",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "change_character",
                "description": "キャラクターを変更する",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "character": {
                            "type": "string",
                            "description": "変更先のキャラクターキーまたは名称（例: metan / 四国めたん / めたん）",
                        }
                    },
                    "required": ["character"],
                },
            },
        },
    ]


def fetch_voicevox_speakers(base_url: str) -> List[dict]:
    res = requests.get(f"{base_url}/speakers", timeout=10)
    res.raise_for_status()
    return res.json()


def find_style_id(speaker: dict) -> int:
    styles = speaker.get("styles") or []
    for style in styles:
        if style.get("name") == "ノーマル":
            return style["id"]
    if styles:
        return styles[0]["id"]
    raise ValueError("VOICEVOX style id が見つかりません。")


def resolve_speaker_ids(
    base_url: str, character_defs: Dict[str, Dict[str, str]]
) -> Dict[str, int]:
    speakers = fetch_voicevox_speakers(base_url)
    speaker_ids: Dict[str, int] = {}
    for key, info in character_defs.items():
        env_key = f"VOICEVOX_SPEAKER_ID_{key.upper()}"
        if env_key in os.environ:
            speaker_ids[key] = int(os.environ[env_key])
            continue
        target = info["voicevox_style"]
        matched = None
        for speaker in speakers:
            if speaker.get("name") == target:
                matched = speaker
                break
        if matched is None:
            for speaker in speakers:
                if target in (speaker.get("name") or ""):
                    matched = speaker
                    break
        if matched is None:
            raise ValueError(f"VOICEVOX話者が見つかりません: {target}")
        speaker_ids[key] = find_style_id(matched)
    return speaker_ids


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


def record_to_temp_wav(seconds: float) -> str:
    import sounddevice as sd
    import soundfile as sf

    print("音声取得中...")
    recording = sd.rec(
        int(16000 * seconds),
        samplerate=16000,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(temp_path, recording, 16000)
    return temp_path


def transcribe_audio(model: WhisperModel, audio_path: str) -> str:
    segments, _info = model.transcribe(audio_path, language="ja")
    texts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def transcribe_with_fallback(
    model: WhisperModel,
    audio_path: str,
    fallback_model: Optional[WhisperModel],
) -> str:
    try:
        return transcribe_audio(model, audio_path)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if fallback_model is None:
            raise
        print("CUDAメモリ不足のため、CPUモデルで再試行します。")
        return transcribe_audio(fallback_model, audio_path)


def play_audio(wav_bytes: bytes) -> None:
    import sounddevice as sd
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, sr)
    sd.wait()


def execute_tool(
    tool_name: str, tool_args: dict, current_character: str
) -> Tuple[str, Optional[str]]:
    if tool_name == "get_character":
        label = CHARACTERS[current_character]["label"]
        return json.dumps(
            {"character_key": current_character, "label": label}, ensure_ascii=False
        ), None
    if tool_name == "list_characters":
        return json.dumps(
            {
                "characters": [
                    {"key": key, "label": info["label"]}
                    for key, info in CHARACTERS.items()
                ]
            },
            ensure_ascii=False,
        ), None
    if tool_name == "change_character":
        target = (tool_args.get("character") or "").strip()
        if target in CHARACTERS:
            new_key = target
        else:
            new_key = None
            for key, info in CHARACTERS.items():
                label = info["label"]
                if target == label or (target and target in label):
                    new_key = key
                    break
            if new_key is None:
                return (
                    f"未対応のキャラクターです: {target}. "
                    f"利用可能: {', '.join(CHARACTERS.keys())}",
                    None,
                )
        label = CHARACTERS[new_key]["label"]
        return json.dumps(
            {"character_key": new_key, "label": label}, ensure_ascii=False
        ), new_key
    return f"未対応のツールです: {tool_name}", None


def choose_tool_choice(user_text: str):
    text = user_text.strip()
    if any(k in text for k in ["キャラ", "キャラクター", "誰", "名前", "一覧", "リスト"]):
        if any(k in text for k in ["一覧", "リスト"]):
            return {"type": "function", "function": {"name": "list_characters"}}
        if any(k in text for k in ["変更", "変えて", "変わって", "チェンジ"]):
            return "auto"
        if any(k in text for k in ["名前", "誰", "キャラ"]):
            return {"type": "function", "function": {"name": "get_character"}}
    return "auto"


def run_agent(
    client: OpenAI,
    model: str,
    tools: List[dict],
    history: List[dict],
    user_text: str,
    current_character: str,
) -> Tuple[str, str]:
    messages: List[dict] = [
        {"role": "system", "content": persona_system_message(current_character)}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    tool_choice = os.environ.get("VLLM_TOOL_CHOICE", "auto")
    if tool_choice == "auto":
        tool_choice = choose_tool_choice(user_text)
    while True:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0.7,
            )
        except BadRequestError as exc:
            if "auto" in tool_choice and "tool choice" in str(exc).lower():
                print(
                    "サーバー側でauto tool choiceが無効のため、"
                    "ツール呼び出しなしで再試行します。"
                )
                tool_choice = "none"
                res = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=0.7,
                )
            else:
                raise
        msg = res.choices[0].message
        tool_calls = msg.tool_calls or []
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                raw = (call.function.arguments or "{}").strip()
                if not raw.startswith("{"):
                    raw = "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = {}
                result, new_character = execute_tool(
                    call.function.name, args, current_character
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )
                if new_character and new_character != current_character:
                    current_character = new_character
                    messages.append(
                        {
                            "role": "system",
                            "content": persona_system_message(current_character),
                        }
                    )
            continue
        return msg.content or "", current_character


def main() -> int:
    args = parse_args()
    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")
    voicevox_url = os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021")
    whisper_model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
    whisper_device = os.environ.get("WHISPER_DEVICE", "cuda")
    whisper_compute = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
    fallback_model_name = os.environ.get("WHISPER_FALLBACK_MODEL", "medium")

    try:
        speaker_ids = resolve_speaker_ids(voicevox_url, CHARACTERS)
    except Exception as exc:
        print(f"VOICEVOX話者の取得に失敗しました: {exc}", file=sys.stderr)
        return 1

    current_character = args.character
    current_speaker_id = speaker_ids[current_character]

    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    tools = build_tools()

    if not args.text:
        model_whisper = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute,
        )
        fallback_whisper: Optional[WhisperModel] = None
        if whisper_device == "cuda":
            fallback_whisper = WhisperModel(
                fallback_model_name, device="cpu", compute_type="int8"
            )
    else:
        model_whisper = None
        fallback_whisper = None

    history: List[dict] = []
    print("会話を開始します。終了するには「終了」と話してください。")
    print(
        f"開始キャラクター: {CHARACTERS[current_character]['label']} "
        f"(speaker_id={current_speaker_id})"
    )

    while True:
        try:
            if args.text:
                user_text = input("あなた: ").strip()
            else:
                temp_path = record_to_temp_wav(args.record_seconds)
                try:
                    user_text = transcribe_with_fallback(
                        model_whisper, temp_path, fallback_whisper
                    )
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                print(f"あなた: {user_text}")
        except KeyboardInterrupt:
            print("\n終了します。")
            break

        if not user_text:
            print("聞き取れませんでした。もう一度お願いします。")
            continue
        if user_text in EXIT_WORDS:
            print("終了します。")
            break

        answer_text, current_character = run_agent(
            client, model, tools, history, user_text, current_character
        )
        current_speaker_id = speaker_ids[current_character]
        print(f"{CHARACTERS[current_character]['label']}: {answer_text}")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer_text})

        if args.no_voice:
            continue
        try:
            wav_bytes = synthesize_voice(
                answer_text, base_url=voicevox_url, speaker=current_speaker_id
            )
            play_audio(wav_bytes)
        except OSError as exc:
            print(f"再生に失敗しました: {exc}", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"音声合成に失敗しました: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
