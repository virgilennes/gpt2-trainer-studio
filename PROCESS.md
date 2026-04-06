# GPT-2 Text Generator: ML Pipeline Process

This document is a didactic walkthrough of the machine learning pipeline implemented in this application. Each section explains **what** is happening, **why** it matters from an AI/ML perspective, and **how** the code implements it. The goal is to build intuition about the core concepts behind training and using a generative language model.

---

## Table of Contents

1. [The Transformer and GPT-2 Architecture](#1-the-transformer-and-gpt-2-architecture)
2. [Tokenization: From Text to Numbers](#2-tokenization-from-text-to-numbers)
3. [Dataset Preparation: Feeding the Model](#3-dataset-preparation-feeding-the-model)
4. [Training Configuration: The Hyperparameter Landscape](#4-training-configuration-the-hyperparameter-landscape)
5. [Training Execution: How the Model Learns](#5-training-execution-how-the-model-learns)
6. [Evaluation: Measuring What the Model Learned](#6-evaluation-measuring-what-the-model-learned)
7. [Text Generation: Sampling from a Probability Distribution](#7-text-generation-sampling-from-a-probability-distribution)
8. [The Full Pipeline Orchestration](#8-the-full-pipeline-orchestration)


---

## 1. The Transformer and GPT-2 Architecture

### What is a Transformer?

The Transformer is a neural network architecture introduced in the 2017 paper "Attention Is All You Need" (Vaswani et al.). Its key innovation is the **self-attention mechanism**, which allows every token in a sequence to "attend to" (i.e., weigh the relevance of) every other token. This replaced the sequential processing of earlier architectures like RNNs and LSTMs, enabling massive parallelization during training.

A Transformer has two main components:

- **Encoder:** reads the full input sequence bidirectionally (used in models like BERT).
- **Decoder:** generates output one token at a time, attending only to previously generated tokens (left-to-right).

### What is GPT-2?

GPT-2 (Generative Pre-trained Transformer 2) is a **decoder-only** Transformer. It was pre-trained by OpenAI on a large web corpus using a **causal language modeling** (CLM) objective: given a sequence of tokens, predict the next token. Because it only looks at tokens to the left (past), it is naturally suited for text generation.

The "small" variant loaded in this application has:

| Property        | Value       |
|-----------------|-------------|
| Layers          | 12          |
| Hidden size     | 768         |
| Attention heads | 12          |
| Parameters      | ~124 million |
| Vocabulary      | 50,257 tokens |

### Why decoder-only?

Encoder-decoder models (like T5) are designed for sequence-to-sequence tasks (translation, summarization). Decoder-only models are optimized for **autoregressive generation**: they produce text one token at a time, each conditioned on all previous tokens. This makes GPT-2 a natural fit for open-ended text generation, story writing, and completion tasks.

### How it is implemented

**Code:** `backend/app/engines/model_loader.py`

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

`AutoModelForCausalLM` is Hugging Face's generic loader for causal (decoder-only) language models. It downloads the pre-trained weights from the Hugging Face Hub and instantiates a `GPT2LMHeadModel`, the GPT-2 architecture with a **language modeling head** (a linear layer that projects the hidden states to vocabulary-sized logits for next-token prediction).

The function extracts a `ModelSummary` from the model config:

```python
summary = ModelSummary(
    name="gpt2",
    num_layers=config.n_layer,        # 12
    num_parameters=sum(p.numel() for p in model.parameters()),  # ~124M
    hidden_size=config.n_embd,        # 768
    vocab_size=config.vocab_size,     # 50257
)
```

The `CacheManager` (`backend/app/engines/cache_manager.py`) wraps Hugging Face's local filesystem cache so the ~500MB model download only happens once.


---

## 2. Tokenization: From Text to Numbers

### What is tokenization?

Neural networks operate on numbers, not text. **Tokenization** is the process of converting raw text into a sequence of integer IDs that the model can process. The choice of tokenization strategy has a profound impact on model performance.

### Three approaches to tokenization

| Strategy           | Example: "unhappiness"       | Pros                          | Cons                          |
|--------------------|------------------------------|-------------------------------|-------------------------------|
| **Character-level** | `u, n, h, a, p, p, i, n, e, s, s` | Tiny vocabulary (~100)       | Very long sequences, hard to learn word meaning |
| **Word-level**      | `unhappiness`                | Intuitive                     | Huge vocabulary, can't handle unseen words |
| **Subword (BPE)**   | `un, happiness`              | Balanced vocabulary, handles rare words | Slightly less intuitive |

### Byte-Pair Encoding (BPE)

GPT-2 uses **Byte-Pair Encoding**, a subword tokenization algorithm. BPE starts with individual characters and iteratively merges the most frequent adjacent pairs into new tokens. The result is a vocabulary of ~50,000 subword units that can represent any text. Common words get their own token, while rare words are split into recognizable pieces.

For example:
- `"Hello"` -> `["Hello"]` (common word, single token)
- `"tokenization"` -> `["token", "ization"]` (split into meaningful subwords)
- `"GPT2"` -> `["G", "PT", "2"]` (rare compound, split further)

This is why GPT-2's vocabulary size is 50,257: it is the number of unique subword tokens the model learned during pre-training.

### Why subword tokenization matters for generation

Subword tokenization gives the model a powerful inductive bias: it can generalize to words it has never seen by composing them from known subword pieces. This is critical for a generative model because it will never encounter an "out-of-vocabulary" word, since any string can be decomposed into known subwords.

### How it is implemented

**Code:** `backend/app/engines/model_loader.py`

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Demonstrate subword breakdown
token_ids = tokenizer.encode("Machine learning is fascinating.")
tokens = tokenizer.convert_ids_to_tokens(token_ids)
# tokens: ['Machine', 'Ġlearning', 'Ġis', 'Ġfasc', 'inating', '.']
```

The `Ġ` prefix represents a leading space. BPE encodes whitespace as part of the token. The word "fascinating" is split into `"Ġfasc"` + `"inating"` because it is not frequent enough to be a single token.

The tokenizer is loaded alongside the model and shared across all pipeline stages (dataset preparation, training, evaluation, generation).


---

## 3. Dataset Preparation: Feeding the Model

### What is WikiText-2?

WikiText-2 is a benchmark language modeling dataset derived from Wikipedia's "Good" and "Featured" articles. It contains approximately 2 million tokens of clean, well-written English text. It is small enough for educational use but large enough to demonstrate real training dynamics.

### Why do we need fixed-length sequences?

Transformers process fixed-length sequences. GPT-2 was pre-trained with a context window of 1024 tokens, but for fine-tuning we use a smaller `block_size` (default: 128 tokens) for memory efficiency. The entire corpus is tokenized into one long sequence of token IDs, then sliced into non-overlapping chunks of exactly `block_size` tokens.

This is a key concept: **the model does not see "sentences" or "paragraphs."** It sees fixed-length windows of token IDs. Each window is an independent training example. The model learns to predict each token given all preceding tokens within that window.

```
Raw text:  "The quick brown fox jumps over the lazy dog. The cat sat..."
Token IDs: [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 383, 3797, 3332, ...]
                    | chunk into blocks of size 4 (simplified) |
Block 1:   [464, 2068, 7586, 21831]
Block 2:   [18045, 625, 262, 16931]
Block 3:   [3290, 13, 383, 3797]
Remainder: [3332]  <-- discarded (does not fill a complete block)
```

### Train/validation split

The dataset comes pre-split into training and validation sets. The training set is what the model learns from; the validation set is held out to measure how well the model generalizes to unseen text. If the model performs well on training data but poorly on validation data, it is **overfitting**: memorizing rather than learning patterns.

### How it is implemented

**Code:** `backend/app/engines/dataset_preparer.py`

```python
class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: Any, block_size: int = 128):
        token_ids = tokenizer.encode(text)
        num_complete_blocks = len(token_ids) // block_size
        total_tokens = num_complete_blocks * block_size
        token_ids = token_ids[:total_tokens]

        self.examples = []
        for i in range(0, total_tokens, block_size):
            self.examples.append(
                torch.tensor(token_ids[i : i + block_size], dtype=torch.long)
            )
```

The `prepare_dataset()` function downloads WikiText-2 via `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")`, filters empty lines, and creates two `TextDataset` instances. It returns a `DatasetStats` object with sample counts, vocabulary size, and block size.


---

## 4. Training Configuration: The Hyperparameter Landscape

### What are hyperparameters?

Hyperparameters are settings that control the training process itself. They are not learned by the model but set by the practitioner. Choosing good hyperparameters is one of the most important (and difficult) parts of training a neural network.

### The hyperparameters in this application

#### Learning Rate (default: 5e-5)

The learning rate controls **how big a step** the optimizer takes when updating the model's weights. Think of training as navigating a hilly landscape where you are trying to find the lowest valley (minimum loss):

- **Too high:** You overshoot the valley and bounce around, never converging. Loss may even increase.
- **Too low:** You inch toward the valley so slowly that training takes forever, and you might get stuck in a shallow local minimum.
- **Just right:** You descend smoothly into a good minimum.

For fine-tuning pre-trained models like GPT-2, learning rates are typically very small (1e-5 to 5e-5) because the model already has good weights. You want to nudge them, not overwrite them.

#### Batch Size (default: 8)

The batch size is **how many training examples the model sees before updating its weights**. Each update step computes the average loss over a batch and adjusts weights accordingly.

- **Larger batches:** More stable gradient estimates, but require more memory and can lead to sharp minima (worse generalization).
- **Smaller batches:** Noisier gradients, which can actually help escape local minima, but training is slower per example.

Batch size 8 is a practical default for fine-tuning on consumer hardware.

#### Number of Epochs (default: 3)

An **epoch** is one complete pass through the entire training dataset. After 3 epochs, the model has seen every training example 3 times.

- **Too few epochs:** The model has not learned enough (underfitting).
- **Too many epochs:** The model memorizes the training data (overfitting). You will see training loss decrease while validation loss starts increasing.

For fine-tuning, 2-5 epochs is typical. The model already knows language from pre-training, so it does not need many passes to adapt.

#### Warmup Steps (default: 0)

Warmup gradually increases the learning rate from 0 to the target value over the first N steps. This prevents the model from making large, destructive updates at the very beginning of training when gradients may be unstable. It is especially useful with large learning rates or when the model has not seen the new data distribution yet.

#### Weight Decay (default: 0.0)

Weight decay is a **regularization** technique that penalizes large weight values by adding a fraction of the weight magnitude to the loss. This encourages the model to use smaller, more distributed weights, which tends to improve generalization. It is the optimizer-level equivalent of L2 regularization.

### The Data Collator

The `DataCollatorForLanguageModeling` with `mlm=False` is a critical detail. Setting `mlm=False` tells the collator to prepare data for **causal language modeling** (predict the next token) rather than **masked language modeling** (predict masked tokens, as in BERT). For a decoder-only model like GPT-2, this is essential. The labels are simply the input shifted by one position:

```
Input:  [The, quick, brown, fox]
Labels: [quick, brown, fox, jumps]
```

The model learns to predict each next token given all previous tokens.

### How it is implemented

**Code:** `backend/app/models/schemas.py` (validation), `backend/app/engines/training_engine.py` (mapping)

```python
# Pydantic model with range validators
class TrainingConfig(BaseModel):
    learning_rate: float = 5e-5   # 1e-6 to 1e-2
    batch_size: int = 8           # 1 to 64
    num_epochs: int = 3           # 1 to 20
    warmup_steps: int = 0         # 0 to 1000
    weight_decay: float = 0.0     # 0.0 to 1.0

# Conversion to Hugging Face TrainingArguments
def config_to_training_args(config, output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        save_strategy="epoch",
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )
```


---

## 5. Training Execution: How the Model Learns

### The training loop, conceptually

Training a neural network is an iterative optimization process. At a high level, each step does:

1. **Forward pass:** Feed a batch of token sequences into the model. The model predicts a probability distribution over the vocabulary for each position.
2. **Loss computation:** Compare the model's predictions to the actual next tokens using **cross-entropy loss**. Cross-entropy measures how far the model's predicted probability distribution is from the true distribution (where all probability mass is on the correct next token).
3. **Backward pass (backpropagation):** Compute the gradient of the loss with respect to every weight in the model. This tells us the direction and magnitude to adjust each weight to reduce the loss.
4. **Weight update:** The optimizer (AdamW, in this case) uses the gradients to update the model's ~124 million parameters.

This cycle repeats for every batch in every epoch.

### Cross-entropy loss for language modeling

For causal language modeling, the loss at each position is:

```
Loss(position) = -log( P(correct_token | previous_tokens) )
```

If the model assigns high probability to the correct next token, the loss is low. If it assigns low probability, the loss is high. The total loss is the average across all positions and all examples in the batch.

Over time, training drives the model to assign higher probability to the correct next tokens, which means it is getting better at predicting (and therefore generating) coherent text.

### AdamW optimizer

This application uses **AdamW** (the default in Hugging Face Trainer), an optimizer that:

- Maintains per-parameter **momentum** (exponential moving average of past gradients) to smooth out noisy updates.
- Maintains per-parameter **adaptive learning rates** (scales the learning rate based on the history of gradient magnitudes) so that frequently-updated parameters get smaller steps.
- Applies **decoupled weight decay**: unlike classic Adam with L2 regularization, AdamW applies weight decay directly to the weights rather than through the gradient, which is theoretically more correct.

### Checkpointing

After each epoch, the Trainer saves a **checkpoint**: a snapshot of the model weights, optimizer state, and training progress. This serves two purposes:

1. **Fault tolerance:** If training crashes, you can resume from the last checkpoint instead of starting over.
2. **Model selection:** You can compare checkpoints from different epochs and pick the one with the best validation performance (the epoch just before overfitting begins).

### Graceful error handling

If training fails (e.g., out-of-memory error, NaN loss), the engine attempts to save an emergency checkpoint before propagating the error. This ensures that partial training progress is never lost.

### How it is implemented

**Code:** `backend/app/engines/training_engine.py`

```python
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MetricsCallback(ws_callback=ws_callback)],
)

train_output = trainer.train()
```

The `MetricsCallback` is a custom `TrainerCallback` that fires on every logging step, constructs a `TrainingMetrics` object (epoch, step, train_loss, learning_rate, elapsed time, ETA), and sends it to the frontend via WebSocket for live visualization.

A module-level `_stop_requested` flag is checked on every step via `on_step_end`. When set (by the `/api/training/stop` endpoint), it sets `control.should_training_stop = True` to halt the Trainer loop cleanly.


---

## 6. Evaluation: Measuring What the Model Learned

### What is perplexity?

**Perplexity** is the standard metric for evaluating language models. Intuitively, it measures **how "surprised" the model is by the text**. Lower perplexity means the model predicts the text more confidently.

Mathematically, perplexity is the exponential of the average cross-entropy loss:

```
Perplexity = exp( average_cross_entropy_loss )
```

A perplexity of 100 means the model is, on average, as uncertain as if it were choosing uniformly among 100 tokens at each position. A perplexity of 20 means it has narrowed down to about 20 plausible tokens per position, which is much better.

### Why evaluate on a validation set?

The validation set contains text the model has never seen during training. Evaluating on it tells us whether the model has learned **generalizable patterns** in language (good) or has merely **memorized the training data** (bad, this is overfitting).

A well-trained model should have:
- Low training loss (it fits the training data)
- Low validation loss / perplexity (it generalizes to new data)
- A gap between training and validation loss that is not too large (not overfitting)

### Baseline comparison

Comparing the fine-tuned model's perplexity against the original (pre-trained) GPT-2 baseline tells us whether fine-tuning actually helped. If the fine-tuned model has lower perplexity on WikiText-2, it means the model has adapted to this specific text domain. The improvement percentage quantifies this:

```
improvement = (baseline_perplexity - trained_perplexity) / baseline_perplexity * 100%
```

### How it is implemented

**Code:** `backend/app/engines/evaluation_engine.py`

```python
def evaluate(model, tokenizer, val_dataset):
    model.eval()  # Disable dropout and other training-specific behaviors
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():  # No gradient computation needed for evaluation
        for batch in dataloader:
            input_ids = batch.to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    return EvalResult(perplexity=perplexity, val_loss=avg_loss)
```

Key details:
- `model.eval()` switches the model to evaluation mode (disables dropout).
- `torch.no_grad()` disables gradient tracking, saving memory and computation since we are not updating weights.
- `labels=input_ids.clone()`: for causal LM, the labels are the input itself (the model internally shifts them to create the next-token prediction task).

The `compare_baseline()` function loads a fresh `AutoModelForCausalLM.from_pretrained("gpt2")`, evaluates both models, and computes the improvement percentage.


---

## 7. Text Generation: Sampling from a Probability Distribution

### How autoregressive generation works

Text generation with GPT-2 is **autoregressive**: the model generates one token at a time, and each new token is fed back as input to generate the next one.

```
Step 1: Input: "The cat"         -> Model predicts next token -> "sat"
Step 2: Input: "The cat sat"     -> Model predicts next token -> "on"
Step 3: Input: "The cat sat on"  -> Model predicts next token -> "the"
...and so on until max_length is reached or an end-of-sequence token is generated.
```

At each step, the model outputs a **logit** (raw score) for every token in the vocabulary (~50,257 values). These logits are converted to probabilities via the **softmax** function. The question is: how do we pick the next token from this distribution?

### Decoding strategies

#### Greedy decoding (not used here)

Always pick the token with the highest probability. This is deterministic and often produces repetitive, boring text because the model gets stuck in high-probability loops ("the the the...").

#### Sampling with temperature

**Temperature** scales the logits before applying softmax:

```
P(token) = softmax( logits / temperature )
```

- **Temperature < 1.0:** Sharpens the distribution. High-probability tokens become even more likely, low-probability tokens become negligible. Output is more focused and predictable.
- **Temperature = 1.0:** Uses the model's raw probabilities as-is.
- **Temperature > 1.0:** Flattens the distribution. All tokens become more equally likely. Output is more random and creative, but also more likely to be incoherent.

#### Top-k sampling

Before sampling, keep only the **k most probable tokens** and redistribute probability among them. This prevents the model from ever selecting very unlikely tokens (which could produce nonsensical text) while still allowing diversity among the top candidates.

- **k = 1:** Equivalent to greedy decoding.
- **k = 50 (default):** The model chooses from its top 50 candidates at each step.

#### Top-p (nucleus) sampling

Instead of a fixed number of tokens, keep the **smallest set of tokens whose cumulative probability exceeds p**. This adapts to the model's confidence:

- When the model is very confident (one token has 95% probability), top-p = 0.9 might keep only 1-2 tokens.
- When the model is uncertain (probability spread across many tokens), top-p = 0.9 might keep 50+ tokens.

This is often more effective than top-k because it adapts to the shape of the distribution rather than using a fixed cutoff.

#### Combining strategies

In practice, temperature, top-k, and top-p are often used together. The model in this application applies all three:

1. Scale logits by temperature.
2. Filter to top-k tokens.
3. Filter to top-p nucleus.
4. Sample from the remaining distribution.

### How it is implemented

**Code:** `backend/app/engines/generation_engine.py`

```python
def generate(model, tokenizer, prompt, params):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=True,          # Enable sampling (not greedy)
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            max_length=params.max_length,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return GenerationResult(text=generated_text, tokens_generated=...)
```

The `compare_generation()` function runs the same prompt through both the baseline and fine-tuned models, returning both outputs for side-by-side comparison. This lets users see how fine-tuning on WikiText-2 changes the model's writing style and coherence.


---

## 8. The Full Pipeline Orchestration

### Pipeline State Machine

The entire process is governed by a **state machine** (`backend/app/pipeline.py`) that enforces the correct ordering of operations:

```
Idle -> Model Loading -> Model Loaded -> Dataset Preparing -> Dataset Ready
     -> Training -> Trained -> Evaluating -> Evaluated -> Generating
```

Each transition has **guards**: you cannot start training without a prepared dataset, you cannot evaluate without a trained model. This prevents the user (or the demo mode) from executing operations out of order, which would cause runtime errors.

State transitions are broadcast to all connected frontend clients via WebSocket, so the UI always reflects the current pipeline stage.

### Automated Demo Mode

The `DemoOrchestrator` (`backend/app/engines/demo_orchestrator.py`) automates the entire pipeline as an educational walkthrough. It executes the six stages in sequence:

1. **Model Loading**: downloads GPT-2 and displays architecture summary
2. **Dataset Preparation**: downloads WikiText-2, tokenizes, and shows statistics
3. **Training Configuration**: sets default hyperparameters with explanations
4. **Training Execution**: runs training with live loss curve visualization
5. **Evaluation**: computes perplexity and compares against baseline
6. **Text Generation**: generates sample text and shows baseline vs. fine-tuned comparison

Between each step, the orchestrator pauses (2s/5s/10s depending on speed setting), sends educational commentary explaining the current step, and highlights the corresponding UI component. Users can pause, resume, or skip steps at any time.

### Real-Time Communication

All long-running operations (model loading, dataset preparation, training, evaluation) stream progress updates to the frontend via a single WebSocket connection (`backend/app/api/websocket.py`). The `ConnectionManager` rate-limits progress messages to intervals of 2 seconds or less to avoid flooding the channel, and sends the current pipeline state to newly connected clients for immediate sync.

### Error Handling

Every engine wraps its operations in structured error handling (`backend/app/api/error_handler.py`). Failures produce an `AppError` with a domain-specific error code (`MODEL_LOAD_FAILED`, `DATASET_DOWNLOAD_FAILED`, `TRAINING_ERROR`, `EVALUATION_ERROR`, `GENERATION_ERROR`) that the middleware converts into a JSON `ErrorResponse` with `error_code`, `message`, and optional `details`. This ensures the frontend always receives actionable error information rather than raw stack traces.

---

## Summary: The ML Pipeline at a Glance

```
+---------------------------------------------------------------------+
|                   GPT-2 Fine-Tuning Pipeline                        |
+---------------------------------------------------------------------+
|                                                                     |
|  1. LOAD MODEL          GPT-2 small (124M params, decoder-only)    |
|     model_loader.py     AutoModelForCausalLM.from_pretrained       |
|                                                                     |
|  2. LOAD TOKENIZER      BPE subword tokenizer (50,257 vocab)       |
|     model_loader.py     AutoTokenizer.from_pretrained              |
|                                                                     |
|  3. PREPARE DATASET     WikiText-2 -> tokenize -> chunk (128 tok)  |
|     dataset_preparer.py TextDataset: fixed-length, non-overlapping |
|                                                                     |
|  4. CONFIGURE TRAINING  lr=5e-5, batch=8, epochs=3, mlm=False      |
|     training_engine.py  TrainingConfig -> HF TrainingArguments     |
|                                                                     |
|  5. TRAIN               Forward -> Loss -> Backward -> Update      |
|     training_engine.py  HF Trainer + MetricsCallback (WebSocket)   |
|                                                                     |
|  6. EVALUATE            Perplexity = exp(avg cross-entropy loss)    |
|     evaluation_engine.py Compare fine-tuned vs. baseline GPT-2     |
|                                                                     |
|  7. GENERATE            Autoregressive sampling: temp/top-k/top-p  |
|     generation_engine.py model.generate() with configurable params |
|                                                                     |
+---------------------------------------------------------------------+
```
