# Tasks

## Task 1: Project Scaffolding and Configuration

- [x] 1.1 Initialize Vite + React + TypeScript frontend project in `frontend/` directory
- [x] 1.2 Install frontend dependencies: shadcn/ui, recharts, zod, fast-check
- [x] 1.3 Initialize Python FastAPI backend project in `backend/` directory with pyproject.toml
- [x] 1.4 Install backend dependencies: fastapi, uvicorn, transformers, torch, datasets, hypothesis, pytest
- [x] 1.5 Create backend directory structure: `backend/app/`, `backend/app/engines/`, `backend/app/models/`, `backend/app/api/`
- [x] 1.6 Create frontend directory structure: `frontend/src/components/`, `frontend/src/hooks/`, `frontend/src/lib/`, `frontend/src/types/`
- [x] 1.7 Configure CORS middleware in FastAPI for frontend dev server

## Task 2: Backend Data Models and Error Handling

- [x] 2.1 Create Pydantic models in `backend/app/models/schemas.py`: TrainingConfig, GenerationParams, ModelSummary, DatasetStats, TrainingMetrics, EvalResult, ComparisonResult, GenerationResult, CompareGenerationResult, ErrorResponse, WSMessage, DemoConfig
- [x] 2.2 Add Pydantic field validators for TrainingConfig ranges (learning_rate 1e-6 to 1e-2, batch_size 1-64, epochs 1-20, warmup_steps 0-1000, weight_decay 0-1)
- [x] 2.3 Add Pydantic field validators for GenerationParams ranges (temperature 0.1-2.0, top_k 1-100, top_p 0.1-1.0, max_length 10-500)
- [x] 2.4 Create structured error handler middleware that catches exceptions and returns ErrorResponse with error_code and message
- [x] 2.5 Write property test: Structured error responses (Property 13) — for any API error, response contains non-empty error_code and message

## Task 3: Frontend Types and Validation

- [x] 3.1 Create TypeScript types in `frontend/src/types/index.ts`: TrainingConfig, GenerationParams, PipelineState, TrainingMetrics, ModelSummary, DatasetStats, EvalResult, ComparisonResult, GenerationResult, WSMessage
- [x] 3.2 Create Zod validation schemas in `frontend/src/lib/validation.ts` for TrainingConfig and GenerationParams with range constraints matching backend
- [x] 3.3 Write property test: Training config validation rejects out-of-range values (Property 2) — for any value outside acceptable range, Zod schema rejects it

## Task 4: Cache Manager

- [x] 4.1 Implement `backend/app/engines/cache_manager.py` with `is_cached(resource)`, `get_cache_path(resource)`, and cache directory management wrapping HF cache
- [x] 4.2 Write property test: Cache idempotence (Property 11) — loading a resource twice uses cache on second call and produces equivalent results

## Task 5: Model and Tokenizer Loading

- [x] 5.1 Implement `backend/app/engines/model_loader.py` with `load_model()` returning ModelSummary and `load_tokenizer()` returning tokenizer info
- [x] 5.2 Use `transformers.AutoModelForCausalLM.from_pretrained("gpt2")` and `AutoTokenizer.from_pretrained("gpt2")` with cache integration
- [x] 5.3 Return ModelSummary with layer count, parameter count, hidden_size, vocab_size extracted from model config
- [x] 5.4 Handle download failures by raising exceptions that map to MODEL_LOAD_FAILED ErrorResponse
- [x] 5.5 Create REST endpoints: `POST /api/model/load`, `GET /api/model/summary`

## Task 6: Dataset Preparation

- [x] 6.1 Implement `backend/app/engines/dataset_preparer.py` with `prepare_dataset(tokenizer, block_size=128)` returning DatasetStats
- [x] 6.2 Implement TextDataset class that tokenizes corpus text, chunks into fixed-length sequences of `block_size` tokens
- [x] 6.3 Download WikiText-2 via `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")` with cache integration
- [x] 6.4 Split into train/validation sets and return DatasetStats with train_samples, val_samples, vocab_size, block_size
- [x] 6.5 Handle download failures by raising exceptions that map to DATASET_DOWNLOAD_FAILED ErrorResponse
- [x] 6.6 Create REST endpoints: `POST /api/dataset/prepare`, `GET /api/dataset/stats`
- [x] 6.7 Write property test: TextDataset fixed-length sequences (Property 1) — for any corpus and block_size, all sequences have exactly block_size tokens
- [x] 6.8 Write property test: Dataset split non-empty (Property 16) — for any successful preparation, both train and val sets are non-empty

## Task 7: Training Engine

- [x] 7.1 Implement `backend/app/engines/training_engine.py` with `start_training(model, dataset, config, callback)` returning TrainingResult
- [x] 7.2 Create TrainingConfig-to-HF TrainingArguments mapping with Data_Collator configured with MLM=False
- [x] 7.3 Implement custom TrainerCallback that streams TrainingMetrics via WebSocket callback
- [x] 7.4 Save checkpoint per epoch in configurable output directory
- [x] 7.5 Implement graceful error handling: on error, save last checkpoint, return TRAINING_ERROR
- [x] 7.6 Create REST endpoints: `POST /api/training/start`, `POST /api/training/stop`
- [x] 7.7 Write property test: Config round-trip to training arguments (Property 3) — for any valid TrainingConfig, converting to TrainingArguments and reading back produces equivalent values
- [x] 7.8 Write property test: Checkpoint saved per epoch (Property 4) — for any N-epoch training run, exactly N checkpoints exist after completion
- [x] 7.9 Write property test: Checkpoint save/load round-trip (Property 12) — for any saved checkpoint, loading it produces a model with identical outputs

## Task 8: Evaluation Engine

- [x] 8.1 Implement `backend/app/engines/evaluation_engine.py` with `evaluate(model, tokenizer, val_dataset)` returning EvalResult
- [x] 8.2 Calculate perplexity as exp(average cross-entropy loss) on validation set
- [x] 8.3 Implement `compare_baseline(model, tokenizer, val_dataset)` returning ComparisonResult with baseline vs trained perplexity
- [x] 8.4 Handle evaluation errors with EVALUATION_ERROR ErrorResponse
- [x] 8.5 Create REST endpoint: `POST /api/evaluation/run`
- [x] 8.6 Write property test: Valid perplexity (Property 5) — for any model and validation set, perplexity is a positive finite number

## Task 9: Generation Engine

- [x] 9.1 Implement `backend/app/engines/generation_engine.py` with `generate(model, tokenizer, prompt, params)` returning GenerationResult
- [x] 9.2 Implement `compare_generation(baseline_model, trained_model, tokenizer, prompt, params)` returning CompareGenerationResult
- [x] 9.3 Apply generation parameters: temperature, top_k, top_p, max_length via `model.generate()`
- [x] 9.4 Handle generation errors with GENERATION_ERROR ErrorResponse
- [x] 9.5 Create REST endpoints: `POST /api/generation/generate`, `POST /api/generation/compare`
- [x] 9.6 Write property test: Generation non-empty output (Property 6) — for any valid prompt and params, output is non-empty and contains the prompt

## Task 10: WebSocket Communication

- [x] 10.1 Implement WebSocket endpoint in `backend/app/api/websocket.py` accepting connections and managing client sessions
- [x] 10.2 Implement WSMessage serialization with type, payload, and timestamp fields
- [x] 10.3 Implement progress broadcasting: send updates at intervals ≤ 2 seconds during long-running operations
- [x] 10.4 Implement state sync on reconnection: send current pipeline state to newly connected clients
- [x] 10.5 Create `frontend/src/hooks/useWebSocket.ts` hook with auto-reconnect (exponential backoff: 1s, 2s, 4s, max 30s)
- [x] 10.6 Write property test: WebSocket update interval (Property 10) — for any consecutive progress messages, time gap ≤ 2 seconds

## Task 11: Pipeline State Machine

- [x] 11.1 Implement pipeline state machine in `backend/app/pipeline.py` with states: idle, model_loading, model_loaded, dataset_preparing, dataset_ready, training, trained, evaluating, evaluated, generating
- [x] 11.2 Implement state transitions with guards (e.g., can't train without dataset, can't evaluate without trained model)
- [x] 11.3 Broadcast state changes via WebSocket
- [x] 11.4 Create REST endpoint: `GET /api/status`

## Task 12: Frontend State Management

- [x] 12.1 Create React context and useReducer for pipeline state in `frontend/src/lib/state.tsx`
- [x] 12.2 Integrate WebSocket hook with state reducer to update state on incoming messages
- [x] 12.3 Store training metrics history for chart rendering
- [x] 12.4 Implement REST API client in `frontend/src/lib/api.ts` with error handling

## Task 13: Control Panel Component

- [x] 13.1 Build `ControlPanel` component with Shadcn/ui form inputs for training config (learning_rate, batch_size, epochs, warmup_steps, weight_decay) with default values
- [x] 13.2 Integrate Zod validation with form — show field-specific error messages on invalid input
- [x] 13.3 Add action buttons: Load Model, Prepare Dataset, Start Training, Run Evaluation
- [x] 13.4 Disable buttons based on pipeline state (e.g., disable Start Training until dataset is ready)
- [x] 13.5 Add tooltips and contextual help text for all configuration fields
- [x] 13.6 Write property test: Config fields have help text (Property 14) — for any config field, tooltip/help text is non-empty

## Task 14: Progress Panel Component

- [x] 14.1 Build `ProgressPanel` component with live loss chart (training + validation loss) using recharts
- [x] 14.2 Display learning rate schedule chart
- [x] 14.3 Display epoch progress bar and estimated time remaining
- [x] 14.4 Build `ModelSummary` sub-component displaying layer count, parameter count, hidden dimensions
- [x] 14.5 Build `DatasetStats` sub-component displaying vocabulary size, sequence lengths, sample count
- [x] 14.6 Build evaluation results display with perplexity value, trend chart, and baseline comparison
- [x] 14.7 Write property test: Display components render required fields (Property 15) — for any valid data, rendered components contain all required information

## Task 15: Commentary Panel Component

- [x] 15.1 Build `CommentaryPanel` component that displays narration text with auto-scroll
- [x] 15.2 Create commentary content for each pipeline stage: model loading, dataset prep, training config, training execution, evaluation, generation
- [x] 15.3 Update commentary based on WebSocket commentary messages and pipeline state changes

## Task 16: Generation Panel Component

- [x] 16.1 Build `GenerationPanel` with prompt input field and parameter controls (temperature, top_k, top_p, max_length) using Shadcn/ui
- [x] 16.2 Implement sample prompt library with selectable preset prompts
- [x] 16.3 Display generated text output area
- [x] 16.4 Implement side-by-side comparison view for baseline vs trained model output
- [x] 16.5 Add parameter tooltips explaining effect of each generation parameter

## Task 17: Connection Status Component

- [x] 17.1 Build `ConnectionStatus` component showing WebSocket connection state (connected/disconnected/reconnecting)
- [x] 17.2 Display reconnection attempts and status in top bar

## Task 18: Responsive Layout

- [x] 18.1 Build main layout with Control_Panel as left sidebar, Progress_Panel as main area, Commentary_Panel at bottom, Generation_Panel at bottom
- [x] 18.2 Implement responsive breakpoints: desktop (≥1024px) and tablet (≥768px) using CSS grid/flexbox
- [x] 18.3 Implement progressive loading indicators for model and dataset operations

## Task 19: Demo Orchestrator

- [x] 19.1 Implement `backend/app/engines/demo_orchestrator.py` with `start_demo(speed, ws_callback)` executing pipeline steps sequentially
- [x] 19.2 Implement speed settings: fast (2s pauses), medium (5s pauses), slow (10s pauses) between steps
- [x] 19.3 Implement `pause()`, `resume()`, `skip_step()` methods with proper state tracking
- [x] 19.4 Send commentary and step highlight messages via WebSocket at each step
- [x] 19.5 Handle step errors: pause demo, send error to Commentary_Panel, allow retry/skip
- [x] 19.6 Create REST endpoints: `POST /api/demo/start`, `POST /api/demo/pause`, `POST /api/demo/resume`, `POST /api/demo/skip`
- [x] 19.7 Write property test: Demo step sequence (Property 7) — for any demo run, steps execute in correct order
- [x] 19.8 Write property test: Demo commentary and highlight (Property 8) — for any demo step, commentary is emitted and correct component is highlighted
- [x] 19.9 Write property test: Demo pause/resume round-trip (Property 9) — pausing stops progression, resuming continues from same step

## Task 20: Demo Frontend Controls

- [x] 20.1 Build `DemoControls` component with start/pause/resume/skip buttons and speed selector (fast/medium/slow)
- [x] 20.2 Implement visual highlighting of active UI component during demo steps
- [x] 20.3 Integrate demo state with pipeline state to disable manual controls during demo mode

## Task 21: FastAPI Application Assembly

- [x] 21.1 Create `backend/app/main.py` assembling all routes, WebSocket endpoint, CORS, and lifespan handler
- [x] 21.2 Wire up all engine instances with shared state (loaded model, tokenizer, dataset, pipeline state)
- [x] 21.3 Add startup/shutdown handlers for cleanup

## Task 22: Integration Testing

- [x] 22.1 Write integration test for full pipeline: model load → dataset prep → train (1 epoch) → evaluate → generate
- [x] 22.2 Write integration test for WebSocket communication: connect, receive progress, disconnect, reconnect, receive state
- [x] 22.3 Write integration test for demo mode: start, verify step sequence, pause, resume, complete
