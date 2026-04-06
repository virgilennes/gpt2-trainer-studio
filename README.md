# GPT-2 Trainer Studio

An interactive web application for fine-tuning GPT-2 on WikiText-2 with a real-time training dashboard. Load the model, prepare the dataset, configure hyperparameters, watch training metrics live, evaluate perplexity, and generate text — all from a browser.

![Pipeline](Demo/GPT%202-%20Recording%202026-04-05%20200246.mp4)

## Features

- Load GPT-2 small (124M parameters) from Hugging Face with local caching
- Tokenize and prepare WikiText-2 train/val splits automatically
- Configure learning rate, batch size, epochs, warmup steps, and weight decay
- Live training metrics streamed via WebSocket (loss curve, ETA)
- Perplexity evaluation with baseline vs. fine-tuned comparison
- Text generation with temperature, top-k, and top-p controls
- Automated demo mode that walks through the full pipeline with commentary
- Pipeline state machine that enforces correct operation ordering

## Stack

| Layer | Tech |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Recharts |
| Backend | FastAPI, Uvicorn, WebSockets |
| ML | PyTorch, Hugging Face Transformers, Datasets, Accelerate |
| Validation | Pydantic v2, Zod |

## Prerequisites

- Python 3.10+
- Node.js 18+
- ~2GB disk space (GPT-2 model + WikiText-2 dataset)

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/virgilennes/gpt2-trainer-studio.git
cd gpt2-trainer-studio
```

### 2. Backend

```bash
pip install fastapi "uvicorn[standard]" transformers torch datasets websockets pydantic "accelerate>=1.1.0"
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

API docs are available at `http://localhost:8000/docs`.

## Usage

Follow the pipeline steps in order using the left control panel:

1. **Load Model** — downloads GPT-2 and the BPE tokenizer (cached after first run)
2. **Prepare Dataset** — downloads WikiText-2 and tokenizes into 128-token blocks
3. **Configure & Start Training** — set hyperparameters and click Start Training
4. **Run Evaluation** — computes perplexity against the baseline model
5. **Generate Text** — enter a prompt and compare baseline vs. fine-tuned output

Or click **Demo** to run the full pipeline automatically with step-by-step commentary.

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI route handlers + WebSocket
│   │   ├── engines/      # ML logic (loader, trainer, evaluator, generator)
│   │   ├── models/       # Pydantic schemas
│   │   ├── main.py       # App assembly + lifespan
│   │   └── pipeline.py   # State machine
│   └── tests/            # Pytest + Hypothesis property tests
└── frontend/
    └── src/
        ├── components/   # React UI components
        ├── hooks/        # useWebSocket
        ├── lib/          # API client, state reducer, validation
        └── types/        # TypeScript types
```

## ML Pipeline Overview

For a detailed walkthrough of the machine learning concepts — tokenization, training loop, perplexity, sampling strategies — see [PROCESS.md](PROCESS.md).

## License

MIT
