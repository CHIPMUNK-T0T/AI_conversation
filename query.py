import os
import sys
from typing import Tuple

import requests
from openai import OpenAI

# 事前にvLLMでQwen3-4B-Instruct-2507-FP8を起動してから実行してください。

def check_whisper() -> Tuple[bool, str]:
    try:
        import faster_whisper  # noqa: F401

        return True, "faster-whisper はインストール済みです。"
    except Exception as exc:
        return False, f"faster-whisper がインストールされていません: {exc}"


def check_vllm(base_url: str, model: str) -> Tuple[bool, str]:
    try:
        res = requests.get(f"{base_url}/models", timeout=5)
        res.raise_for_status()
    except requests.RequestException as exc:
        return False, f"LLM(vLLM) に通信できません。起動できません: {exc}"

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
        chat = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "疎通確認です。"}],
            temperature=0.0,
        )
        if not chat.choices or not chat.choices[0].message.content:
            return False, "LLM(vLLM) の応答が空です。起動できません。"
    except Exception as exc:
        return False, f"LLM(vLLM) に通信できません。起動できません: {exc}"

    return True, "LLM(vLLM) は通信できました。"


def check_voicebox(base_url: str) -> Tuple[bool, str]:
    try:
        res = requests.get(f"{base_url}/speakers", timeout=5)
        res.raise_for_status()
    except requests.RequestException as exc:
        return False, f"Voicebox(VOICEVOX) に通信できません。起動できません: {exc}"
    return True, "Voicebox(VOICEVOX) は通信できました。"


def main() -> int:
    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")
    voicevox_url = os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021")

    ok = True
    whisper_ok, whisper_msg = check_whisper()
    print(whisper_msg)
    ok = ok and whisper_ok

    vllm_ok, vllm_msg = check_vllm(base_url, model)
    print(vllm_msg)
    ok = ok and vllm_ok

    voice_ok, voice_msg = check_voicebox(voicevox_url)
    print(voice_msg)
    ok = ok and voice_ok

    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
