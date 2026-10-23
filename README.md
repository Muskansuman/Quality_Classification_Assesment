# 8x8 Class Quality Assessment

AI-powered prototype for the **Humaniti AI Build Challenge**: assess an online class recording as **Poor**, **Average**, **Good**, or **Excellent**, with evidence-based observations.

## Approach

- **Pretrained models only** (no custom training).
- **Transcript**: WebVTT if provided, otherwise **faster-whisper** on audio extracted via **ffmpeg**.
- **Features**: talk ratios, questions, homework mentions, silence gaps, opening/closing excerpts.
- **Assessment**: **Ollama** (open-source, local) by default, or optional **OpenAI** cloud. Rubric aligned to organiser feedback. Heuristic fallback if the LLM is unreachable.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on PATH)
- [Ollama](https://ollama.com/) for open-source LLM (recommended), *or* OpenAI API key

## Setup

```bash
cd /home/hitech/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Open-source LLM (Ollama)

```bash
# Install Ollama from https://ollama.com then:
chmod +x scripts/setup_ollama.sh
./scripts/setup_ollama.sh
```

This checks that Ollama is running and pulls **`qwen2.5:7b-instruct`** (good JSON / instruction following on CPU).

Default `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434/v1
```

For **OpenAI** instead, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`.

## Run the app

```bash
streamlit run app.py
```

1. Upload an **MP4** class recording.
2. Optional: upload matching **.vtt** to skip Whisper (faster for sample data).
3. Click **Analyze class**.

For bundled samples, use the **Evaluate local sample files** section at the bottom of the app (reads files from the dataset folder on disk — do not commit recordings to GitHub).

## Batch eval (dev, uses VTT only)

```bash
source .venv/bin/activate
# Uses LLM_PROVIDER from .env (Ollama by default)
python scripts/eval_local.py
```

## Privacy

- Do **not** commit class recordings or transcripts to the repository.
- Uploads are written to a temporary directory and deleted after analysis.
- Redact API keys and personal data in submitted AI interaction logs.

## Deploy

Suitable hosts: [Streamlit Community Cloud](https://streamlit.io/cloud), Hugging Face Spaces, Render.

For cloud deploy with Ollama you need a host that runs Ollama, or switch to `LLM_PROVIDER=openai` and set `OPENAI_API_KEY`.

## Limitations

- Long classes (~45–90 min) make CPU Whisper slow; prefer VTT when testing.
- Without video analysis, AV issues (face not visible, missing background) are inferred mainly from transcript cues and LLM reasoning.
- Whisper segments may lack speaker names; VTT from the platform is richer.
- The built-in **heuristic fallback** is only a demo; use **Ollama** or **OpenAI** for rubric-aligned assessments.

## License

MIT (application code). Sample recordings remain confidential per challenge rules.
