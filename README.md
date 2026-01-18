# AI_conversation

Whisperで音声を文字起こし → LLMで応答生成 → VOICEVOX(Voicebox)で音声合成まで行う会話アプリです。実装は `main.py` にあります。

## 必要なもの
- Python 3.10 以降
- vLLM (LLMサーバー) の起動
- VOICEVOX Engine (Voicebox) の起動
- マイク入力を使う場合は `sounddevice` と `soundfile` が動作する環境

## セットアップ
```bash
python -m venv venv
source ./venv/bin/activate  # Ubuntuの場合
pip install -r requirements.txt
```

## Voicebox(VOICEVOX Engine) 起動例
```bash
sudo docker pull voicevox/voicevox_engine:cpu-latest

sudo docker run -d \
  --name voicevox_engine \
  -p 50021:50021 \
  voicevox/voicevox_engine:cpu-latest
```

## vLLM 起動例
`main.py` で `tool_choice="auto"` を使うには、vLLM側で auto tool choice を有効化する必要があります。

例（オプション名は vLLM のバージョンにより異なるため、実際の環境に合わせて調整してください）:
```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --quantization fp8 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

vLLM側の機能が無効な場合は、環境変数でツールを無効化できます:
```bash
export VLLM_TOOL_CHOICE=none
```

## 使い方 (main.py)

デフォルト設定（ずんだもん、録音5秒）で起動します:
```bash
python main.py
```

### オプション
- `--record-seconds FLOAT`: 録音時間を秒単位で指定 (default: 5.0)
- `--character {zundamon,metan,tsumugi}`: 開始キャラクターを指定 (default: zundamon)
- `--text`: 音声入力の代わりにテキスト入力を使用
- `--no-voice`: 音声合成・再生を行わない

### 実装の仕組み
`main.py` は以下のフローで動作します:
1. **音声取得**: マイクから録音 (`sounddevice`) またはテキスト入力。
2. **音声認識**: `faster_whisper` で音声をテキスト化 (GPU推奨、失敗時はCPUフォールバック機能あり)。
3. **LLM推論**: OpenAI互換API (vLLM) を呼び出し。
    - **Tool Use**: 会話中でキャラクター変更などが要求された場合、`change_character` などのツールを実行してコンテキストを動的に切り替えます。
4. **音声合成**: VOICEVOX Engine API を叩いて音声を生成。
5. **再生**: 生成された音声を再生。

キャラクター定義（`CHARACTERS` 辞書）には、VOICEVOXのスタイル名やペルソナ（システムプロンプト）が含まれており、ここを拡張することで新しいキャラクターを追加できます。



## 単発の会話による動作確認 (single_conversation.py)
音声ファイル指定か、マイク録音で動作します。
```bash
# マイク録音で実行
python single_conversation.py

# 音声ファイルを指定
python single_conversation.py ./sample.wav
```

主なオプション:
- `--record-seconds` (録音秒数)
- `--voicevox-speaker` (話者ID)

## 動作確認 (query.py)
各要素の疎通確認を個別に行います。通信できない場合は「起動できません」「インストールされていません」を表示します。
```bash
python query.py
```

環境変数で接続先を変更できます:
- `VLLM_BASE_URL` (例: http://127.0.0.1:8000/v1)
- `VLLM_MODEL` (例: Qwen/Qwen3-4B-Instruct-2507-FP8)
- `VLLM_API_KEY` (任意)
- `VOICEVOX_URL` (例: http://127.0.0.1:50021)
