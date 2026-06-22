# TakeMeter — Soccer Discourse Classifier

A fine-tuned text classifier that evaluates discourse quality in r/soccer posts and comments, distinguishing between analytical arguments, hot takes, and emotional reactions.

---

## Community Choice

**Community:** r/soccer (reddit.com/r/soccer)

r/soccer is one of the largest sports communities on Reddit, with active discussion spanning match reactions, tactical analysis, transfer news, and fan culture. The discourse quality varies enormously — a single thread can contain a stat-backed tactical breakdown, a bold unprovoked opinion, and a pure emotional reaction side by side. This variation makes it an ideal classification target: the distinctions are real, they matter to community members, and they are frequent enough to collect 200+ labeled examples without difficulty.

---

## Label Taxonomy

### `analysis`
Posts that make a structured argument backed by specific statistics, tactical observation, or historical comparison. Evidence is concrete and verifiable. The post reasons toward a conclusion rather than simply asserting one.

**Example 1:** "Belgium had 23 shots vs IR Iran, their most without scoring in a FIFA World Cup game since 1994 vs Saudi Arabia (28). Their players have fired in 69 shots at the finals since a Belgium player last got on the scoresheet."

**Example 2:** "Eloy Room recorded 15 saves against Ecuador, the most on record since 1966 by any goalkeeper in a FIFA World Cup match without extra time."

---

### `hot_take`
A bold, confident opinion stated without meaningful supporting evidence. The post asserts a claim — often provocative or hyperbolic — rather than arguing for it. The opinion might be true, but the post makes no real attempt to justify it.

**Example 1:** "The Belgian team is a total disgrace year after year."

**Example 2:** "Albert Riera: 'The level of the Bundesliga disappointed me a bit. The teams in the bottom third all play in the exact same way. If Celje were to play against Frankfurt, they could keep up.'"

---

### `reaction`
An immediate emotional response to a specific match event, result, goal, or moment. The post expresses a feeling rather than making an argument. Little to no analytical content — the focus is on the emotional or human dimension of what just happened.

**Example 1:** "Curaçao manager Dick Advocaat (78) in tears after helping Curaçao, the smallest nation to ever make the World Cup, earn a point against Ecuador."

**Example 2:** "My earliest memory is my dad carrying me on his shoulders after we played Algeria in the qualifiers. Now I've gotten to witness Egypt win for the first time in the world's greatest footballing stage."

---

## Data Collection

**Source:** r/soccer — posts and top-level comments from active threads during the 2026 FIFA World Cup group stage (June 2026). Threads sampled included match threads, post-match stat posts, translated news/quote posts, and tactical discussion threads.

**Collection method:** Manual copy-paste from r/soccer into a CSV file. Image posts, memes, and posts without substantive text were excluded — they lack sufficient signal and do not fit any of the three labels cleanly.

**Labeling process:** An AI tool (Claude) was used to pre-label batches of raw comments using the label definitions from planning.md. Every pre-assigned label was reviewed and corrected where necessary before inclusion in the final dataset. This workflow is disclosed in the AI Usage section below.

**Label distribution:**

| Label | Count | Percentage |
|---|---|---|
| `reaction` | 77 | 38.5% |
| `analysis` | 71 | 35.5% |
| `hot_take` | 52 | 26.0% |
| **Total** | **200** | |

### Difficult-to-label examples

**Case 1 — de Jong's through-ball quote:**
> "Many people don't understand anything about football. They watch it, but they don't understand it. I hear people say I don't play through balls, but that's not true. They're simply not paying attention to the game."

This reads like either a hot take (confident assertion without evidence) or a reaction (defensive emotional response to criticism). Decision: labeled `hot_take` because the post asserts a claim about on-pitch performance without providing specific data — it's an argument by authority rather than evidence.

**Case 2 — Schlotterbeck injury report:**
> "Germany's medical staff strongly suspect Nico Schlotterbeck has torn an ankle ligament after an initial stadium examination."

This is factual reporting grounded in professional assessment, which points toward `analysis`. However, it is not making a structured argument — it is reporting a fact. Decision: labeled `analysis`, though this is a borderline case. In retrospect, news reports without argument structure would be better excluded from the dataset.

**Case 3 — Thierry Henry pundit quote:**
> "The team needs to score, not you need to score."

This sits between `hot_take` (a provocative opinion) and `reaction` (an immediate behavioral observation after a match). Decision: labeled `reaction` because it functions as an immediate in-the-moment observation about a player's mindset rather than a structured argument or unsupported claim about player value.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

DistilBERT is a distilled version of BERT — 40% smaller and 60% faster while retaining 97% of BERT's language understanding. It is well-suited for short text classification tasks on small datasets.

**Training setup:**
- Framework: HuggingFace `transformers` + `Trainer` API
- Dataset split: 70% train (140), 15% validation (30), 15% test (30)
- Stratified split to preserve label distribution across splits

**Hyperparameters:**

| Parameter | Value | Reasoning |
|---|---|---|
| `num_train_epochs` | 3 | Conservative choice for 140 training examples — more epochs risk overfitting |
| `learning_rate` | 2e-5 | Standard starting point for fine-tuning BERT-family models |
| `per_device_train_batch_size` | 16 | Fits T4 GPU comfortably; larger batches would reduce gradient noise but memory-constrain |
| `weight_decay` | 0.01 | Light regularization to reduce overfitting on small dataset |
| `warmup_steps` | 50 | Gradual LR warmup stabilizes early training |

**Training results by epoch:**

| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|---|---|---|---|
| 1 | — | 1.097 | 26.7% |
| 2 | 1.107 | 1.083 | 46.7% |
| 3 | 1.092 | 1.060 | 56.7% |

---

## Baseline Description

**Model:** `llama-3.3-70b-versatile` via Groq API (zero-shot)

**Prompt used:**
```
You are classifying soccer posts from r/soccer into exactly one of these three categories.

analysis: Makes a structured argument using specific statistics, tactical observation, or historical comparison. Evidence is concrete and verifiable.
Example: "Eloy Room recorded 15 saves against Ecuador, the most on record since 1966 by any goalkeeper in a FIFA World Cup match without extra time."

hot_take: A bold opinion stated confidently without supporting evidence. Asserts rather than argues.
Example: "The Belgian team is a total disgrace year after year."

reaction: An immediate emotional response to a match, goal, or event. Little to no argument.
Example: "I cannot believe what I just watched. That last-minute winner is going to live in my head forever."

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction
```

All 30 test examples received parseable responses (0 unparseable).

---

## Evaluation Report

### Overall accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq llama-3.3-70b) | **80.0%** |
| Fine-tuned DistilBERT | **53.3%** |

Fine-tuning resulted in a **26.7 percentage point regression** relative to the baseline.

### Per-class metrics — Baseline (Groq)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.90 | 0.82 | 0.86 | 11 |
| `hot_take` | 0.75 | 0.75 | 0.75 | 8 |
| `reaction` | 0.75 | 0.82 | 0.78 | 11 |
| **macro avg** | 0.80 | 0.80 | 0.80 | 30 |

### Per-class metrics — Fine-tuned DistilBERT

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.45 | 0.91 | 0.61 | 11 |
| `hot_take` | 0.00 | 0.00 | 0.00 | 8 |
| `reaction` | 0.75 | 0.55 | 0.63 | 11 |
| **macro avg** | 0.40 | 0.48 | 0.41 | 30 |

### Confusion matrix — Fine-tuned DistilBERT

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|---|---|---|---|
| **True: analysis** | 10 | 0 | 1 |
| **True: hot_take** | 7 | 0 | 1 |
| **True: reaction** | 5 | 0 | 6 |

The model predicted `hot_take` **zero times** across all 30 test examples. All 8 true `hot_take` examples were misclassified — 7 as `analysis` and 1 as `reaction`.

### Wrong predictions — analysis

**Wrong prediction 1:**
> Text: "He was at AZ twice too"
> True: `analysis` | Predicted: `reaction` (confidence: 0.35)

This short factual statement about Advocaat's career was labeled `analysis` because it provides a specific verifiable fact. The model predicted `reaction` — likely because the text is extremely short and lacks the syntactic structure of an analytical post. The model appears to have learned length and complexity as proxies for `analysis`, which this example violates.

**Wrong prediction 2:**
> Text: "The Dutch can make a deep run this year, semi-final might be too conservative even. They have all the right pieces."
> True: `hot_take` | Predicted: `analysis` (confidence: 0.35)

This is a tournament prediction with no supporting evidence — a clear `hot_take`. The model predicted `analysis`, likely because the post discusses team composition and tactics using soccer-specific vocabulary ("deep run," "right pieces"). The model confused topic domain (soccer tactics) with analytical structure.

**Wrong prediction 3:**
> Text: "Resetting your plus 4 GD in the 2nd match is almost impressive lmao"
> True: `reaction` | Predicted: `analysis` (confidence: 0.36)

This sarcastic emotional reaction references a specific statistic (GD). The model predicted `analysis` because it detected a number, but the post's purpose is clearly comedic/emotional — the stat is deployed as punchline, not evidence. The model did not learn to read intent or tone.

### Sample classifications

| Post (excerpt) | True label | Predicted | Confidence | Notes |
|---|---|---|---|---|
| "Belgium had 23 shots vs IR Iran, their most without scoring since 1994 vs Saudi Arabia (28)..." | analysis | analysis | 0.71 | ✅ Correct — stat-heavy post with historical comparison |
| "The Belgian team is a total disgrace year after year" | hot_take | analysis | 0.38 | ❌ Bold opinion predicted as analysis |
| "I just feel so bad for their fans. The golden gen is done." | reaction | reaction | 0.64 | ✅ Correct — clear emotional response |
| "My goat does it again like in 2018 when he saved CR7s penalty" | reaction | analysis | 0.38 | ❌ Historical reference triggered analysis prediction |
| "4 wins out of 14 games, worst point average by an Eintracht coach since 2016" | analysis | analysis | 0.68 | ✅ Correct — specific stat used to make an argument |

The correctly predicted `analysis` example ("Belgium had 23 shots...") is reasonable because the post leads with a specific, verifiable statistic and provides historical comparison — exactly the structural pattern DistilBERT was exposed to most during training.

### What the model learned vs. what I intended

I intended the model to learn the *purpose* of a post: is it making an argument, asserting an opinion, or expressing a feeling? What it actually learned was much shallower: posts containing numbers, player names, and tactical vocabulary get labeled `analysis`; everything else gets labeled `reaction`. The `hot_take` label essentially vanished — the model never learned to distinguish "asserting without evidence" from "arguing with evidence," because both use the same soccer vocabulary.

This is a labeling and data problem as much as a model problem. Many `hot_take` examples in the dataset are short, use soccer-specific terms, and structurally resemble short `analysis` posts — the difference is in intent and evidential status, which is hard to learn from 52 examples. A larger dataset with more diverse `hot_take` examples, or cleaner boundary cases, would give the model more signal to work with.

The confidence scores are also uniformly low (0.34–0.39 on wrong predictions, 0.61–0.71 on correct ones) — suggesting the model is uncertain across the board, not confidently wrong. This is consistent with a model that learned weak surface features rather than meaningful distinctions.

---

## Reflection: Model Learned vs. Intended

The fine-tuned model collapsed toward predicting `analysis` for most inputs. The intended decision boundary was about argumentative structure and evidential status — does the post provide verifiable evidence, assert without support, or express an emotion? The actual decision boundary the model learned was closer to: does the post contain soccer statistics and player names (→ `analysis`) or short emotional language (→ `reaction`)?

The `hot_take` label was the hardest because it sits in between: hot takes often use soccer vocabulary and can reference specific players, but don't actually provide evidence. With only 52 training examples and significant surface overlap with `analysis`, the model never learned this distinction. The baseline (Groq) handled `hot_take` much better (F1: 0.75) because a large language model can reason about argumentative intent from its pretraining — DistilBERT, fine-tuned on 52 examples, cannot.

---

## Spec Reflection

**One way the spec helped:** The spec's emphasis on label design before data collection was the single most valuable structural constraint. Writing decision rules for edge cases (the "stat-supported hot take" and "emotion plus one stat" cases) before annotating 200 examples prevented mid-annotation label drift and gave me consistent criteria to apply throughout.

**One way implementation diverged:** The spec assumed the fine-tuned model would outperform the zero-shot baseline, framing the baseline as something to beat. In this project, the opposite happened — the baseline significantly outperformed the fine-tuned model. This is a valid and informative result: with only 52 `hot_take` training examples and significant surface overlap between `hot_take` and `analysis`, DistilBERT lacked the signal to learn the hardest distinction. A large LLM with pretraining on billions of tokens can reason about argumentative intent; a small model fine-tuned on 140 examples cannot. The divergence from the spec's implied outcome is itself the most important finding of this project.

---

## AI Usage

**Instance 1 — Dataset labeling assistance:**
I directed Claude to label batches of raw r/soccer comments using the label definitions from planning.md. Claude produced a labeled CSV for each batch. I reviewed every pre-assigned label individually and corrected cases I disagreed with — particularly short factual statements (which Claude often labeled `analysis` when I considered them `reaction`) and sarcastic posts (which Claude sometimes labeled `hot_take` when context made them clearly `reaction`). Approximately 10–15% of labels were corrected during review. Pre-labeled examples are not separately flagged in the CSV, but all labels reflect my final judgment after review.

**Instance 2 — Failure pattern analysis:**
After receiving the 14 wrong predictions from Section 4, I pasted them into Claude and asked it to identify common patterns. Claude identified that the model was defaulting to `analysis` for most errors and that `hot_take` predictions were almost entirely absent — matching what the confusion matrix confirmed. I verified this by re-reading all 14 wrong predictions myself before writing the evaluation report. Claude also suggested that posts with any numerical content were being captured by the `analysis` label regardless of purpose, which I verified against examples #3 ("Resetting your plus 4 GD") and #9 ("My goat does it again like in 2018").

**Instance 3 — planning.md generation:**
I provided Claude with my community choice (r/soccer), four real examples from the subreddit, and my initial label ideas. Claude generated a full planning.md draft. I reviewed and adjusted the edge case decision rules and success criteria based on my own annotation experience.

---

*Dataset: `dataset.csv` (200 labeled examples)*
*Evaluation outputs: `evaluation_results.json`, `confusion_matrix.png`*
