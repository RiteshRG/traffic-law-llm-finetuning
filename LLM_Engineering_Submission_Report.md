# CSI 6007: Advanced AI Core Lab
## Open Book Lab Exam: Article-Based AI Assignments

# LLM Engineering Submission Report
## Task 2 — Domain-Specific Fine-Tuned LLM using SFT + QLoRA

### Assignment A: Road Safety — “Sense & Licence” (TOI Editorial)

**Goa Business School, Goa University**  
**M.Sc. Artificial Intelligence**

**Name:** Ritesh Gawde  
**Roll / Seat Number:** 25P0620014  
**Date:** 27 October 2026

---

## 1. Introduction and Problem Context

The assigned “Sense & Licence” editorial raises concerns about road deaths, driver behaviour and enforcement of traffic rules. This project addresses the information-access side of that problem: a small language-model assistant was adapted to answer common Indian traffic-law and road-safety questions in plain language. It is an educational prototype, not a substitute for official legal guidance or emergency services.

## 2. Task Objective

The objective was to adapt `Qwen/Qwen2.5-1.5B` to answer Indian traffic-rule and road-safety questions using supervised fine-tuning (SFT) with parameter-efficient LoRA adapters and 4-bit quantized base weights (QLoRA). The project prepared a small question-answer dataset, formatted and tokenized it for Qwen, trained an adapter, evaluated base and adapted models, saved the adapter, and defined a Gradio question-answer demo.

The training artifacts and notebook outputs support that fine-tuning and evaluation were carried out. The qualitative outputs also show a major remaining generation-quality problem: answers can begin appropriately and then continue with garbled, repeated, or unrelated tokens.

## 3. Technology Stack

| Component | Evidence-supported technology or value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B`, causal language model; approximately 1.5 billion parameters |
| Deep-learning framework | PyTorch; recorded version `2.11.0+cu130` |
| Model and tokenizer | Hugging Face Transformers; recorded version `5.21.0` |
| Dataset handling | Hugging Face Datasets; recorded version `5.1.0` |
| Parameter-efficient tuning | PEFT / LoRA; recorded version `0.21.2` |
| SFT trainer | TRL `SFTTrainer`; recorded version `1.16.0` |
| Quantization | bitsandbytes 4-bit NF4 with double quantization |
| Training optimization | Unsloth installed and invoked; recorded version `2026.10.9` |
| Compute | Notebook output reports Google Colab Tesla T4, with 14.56 GB total GPU memory measured by PyTorch |
| Demo | Gradio interface defined and launched from the notebook; a temporary public URL was printed at runtime |
| Additional judge | GPT-OSS 120B through Groq and LangChain, used on eight answer comparisons |

The notebook output notes that installed `xformers` was built for a different PyTorch/CUDA version and could not load; PyTorch attention was used instead. The 14.56 GB figure is the notebook’s measured device memory, not a generic T4 specification.

## 4. Dataset and Data Understanding

The source was `Pravincoder/Indian_traffic_law_QA`, loaded from Hugging Face. The raw training split contained **200 examples** and a single column named `text`. Each value combined a question and answer using `###Human:` and `###Assistant:` markers; it did not provide separate instruction and response columns. The examples were short English traffic-law and road-safety question-answer pairs.

This content is relevant to the intended assistant because it directly covers questions about traffic rules and safe road use. It is nevertheless a small dataset and the available artifacts do not establish that its answers were legally reviewed or comprehensive.

## 5. Data Preprocessing

The notebook applied a regular expression to parse the combined text field into an `instruction` and a `response`. Rows that did not match the expected marker pattern were assigned missing values and filtered out; both resulting fields also had to be non-empty. A whitespace-cleaning function collapsed repeated whitespace and newlines to single spaces and trimmed the ends. The notebook checked for leftover `###` markers.

The recorded duplicate-removal output reports 200 rows before deduplication, 162 rows afterward, and 38 exact question-answer pairs removed. Two questions still occurred twice with different associated rows, so both instances were retained rather than arbitrarily selecting one answer. A seeded shuffle (`seed=42`) preceded the split. The notebook examined sequence lengths without truncation and selected `max_length=63`; the longest observed example was 63 tokens, so the recorded tokenization did not truncate any examples.

**Figure 1. Dataset preprocessing, duplicate removal and train/validation/evaluation split.**

> **Screenshot insertion placeholder:** Add the provided `load data.png` and `split data.png` images here. These filenames were supplied by the student but were not present in the workspace when this report was prepared, so they could not be visually checked or embedded.
>
> Figure 1 should document the dataset loading/cleaning and final split. The recorded notebook output shows the 200-to-162 deduplication and the resulting split sizes; the screenshot should make those steps visible.

## 6. Train / Validation / Evaluation Split

The cleaned set was shuffled with seed 42, then split into 80% training and 20% temporary data. The temporary portion was split equally between validation and evaluation, each using seed 42.

| Split | Examples | Approximate share of 162 |
|---|---:|---:|
| Training | 129 | 79.63% |
| Validation | 16 | 9.88% |
| Evaluation | 17 | 10.49% |
| **Total** | **162** | **100%** |

The validation set was used for step-based checks during training; the held-out evaluation split was evaluated separately after training. The notebook’s overlap check found **0 of 16** validation questions and **1 of 17** evaluation questions also present in training. This repeated-question overlap can make the final evaluation somewhat optimistic. The small evaluation size also makes the reported estimates sensitive to the particular split.

## 7. Model Architecture and QLoRA Configuration

The base model is the 1.5-billion-parameter Qwen2.5 causal language model, which predicts the next token in a sequence. Training examples were formatted as Qwen chat turns: a user message containing the question followed by an assistant message containing the answer.

The notebook loaded the base model in 4-bit using NF4 quantization, selected FP16 as the 4-bit compute dtype, enabled double quantization, and loaded with automatic device placement. This reduces the memory required for base weights, making adapter training feasible on limited GPU memory. The quantized model was prepared for k-bit training before attaching LoRA.

| LoRA setting | Recorded value |
|---|---|
| Rank (`r`) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | `none` |
| Task type | `CAUSAL_LM` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 18,464,768 of 1,562,179,072 total (1.1820%) |

LoRA adds trainable low-rank matrices to selected attention and MLP projections while leaving the base weights frozen. This means the experiment updates a small fraction of the full model rather than performing full-parameter fine-tuning. The saved adapter configuration independently confirms the base model and LoRA settings.

## 8. Training Configuration

The executed `SFTConfig` code and training output are the primary evidence for the run configuration.

| Parameter | Executed value |
|---|---|
| Epochs | 5 |
| Per-device train batch size | 1 |
| Per-device evaluation batch size | 1 |
| Gradient accumulation | 4 steps |
| Effective batch size | 4 examples per optimizer update (single-device run) |
| Learning rate | `2e-4` |
| Warmup steps | 5 |
| Precision | FP16 enabled |
| Gradient checkpointing | Enabled |
| Maximum sequence length | 63 tokens |
| Logging interval | Every 5 steps |
| Evaluation strategy | Every 10 steps |
| Checkpoint strategy | Save every 20 steps; retain at most 2 |
| Optimizer | Not explicitly set in the notebook configuration; library default applies |
| External experiment reporting | Disabled (`report_to="none"`) |

**Important epoch discrepancy:** Several notebook Markdown cells say the run uses exactly one epoch, and `evaluation_metrics.json` plus the final-results summary also record/print `epochs: 1`. However, the executed `SFTConfig` sets `num_train_epochs=5`; the notebook’s training output shows `[165/165, Epoch 5/5]`, and `checkpoint-165/trainer_state.json` records epoch 5.0. Therefore, this report treats **five epochs** as the actual run and identifies the one-epoch values as stale or inconsistent metadata. The saved metrics JSON should not be used as evidence for the actual epoch count.

## 9. Fine-Tuning Process

The pipeline loaded the base Qwen model and tokenizer, configured 4-bit quantization, prepared the model for k-bit training, attached LoRA adapters, and used the tokenized training and validation splits with TRL’s `SFTTrainer`. The trainer ran for five epochs and wrote checkpoints. The final adapter and tokenizer were saved under `final_model/`; the adapter was not merged into the full base model. The notebook subsequently evaluated the held-out evaluation split and loaded the base model with the saved adapter for generation comparisons.

SFT provides the learning objective: train the assistant to continue formatted question-answer examples. QLoRA is the memory-saving setup that keeps the base model quantized and frozen while training the small LoRA parameter set.

## 10. Evaluation Methodology

**Evaluation loss and perplexity.** Evaluation loss measures next-token prediction error on the formatted evaluation examples. Perplexity is calculated as $\exp(\text{evaluation loss})$; lower indicates a closer fit to the evaluated text. The notebook compares the base model and fine-tuned model on the same evaluation split. These metrics measure fit to the formatted text, not legal factual correctness.

**Qualitative comparison.** The notebook generated answers for traffic-law questions with the original model and with the adapter enabled. The saved comparison CSV contains examples of these responses, including malformed output in both versions.

**Model-based judge.** The notebook used GPT-OSS 120B through Groq, via LangChain structured output, to score eight base/fine-tuned answer pairs. It scored correctness, relevance, and quality (with additional written comparison criteria) and selected a better answer or tie. This is an auxiliary model-based judgment, not a ground-truth legal review or independent legal authority.

## 11. Results

| Metric | Base model | Fine-tuned model |
|---|---:|---:|
| Evaluation loss | 4.5090 | 1.2387 |
| Perplexity | 90.83 | 3.45 |
| Mean LLM-judge overall score (1–10) | 1.12 | 4.00 |
| Judge selections | 0 wins | 7 wins |

Of the eight judge comparisons, one was a tie; the fine-tuned model was selected as better in seven and the base model in zero. The substantially lower loss and perplexity show a stronger fit to this small formatted evaluation set. They do not prove improved legal accuracy or generalization. The judge scores suggest that the adapted answers were often more relevant than the base outputs, but the average fine-tuned score of 4.00 and the recorded response defects show that overall answer quality remained limited.

**Qualitative examples from the saved comparison artifacts:**

| Question | Base model output | Fine-tuned model output |
|---|---|---|
| Can I use a mobile phone while driving in India? | The saved CSV begins with repeated corrupted/CJK tokens and does not provide a usable answer. | Begins: “Using a mobile phone while driving is prohibited, and violators can face fines and penalties.” It then continues with corrupted and unrelated tokens. |
| What should I do when approaching a pedestrian crossing? | The saved CSV contains unrelated corrupted text and an unrelated multimeter question. | Begins: “When approaching a pedestrian crossing, you should slow down and stop to allow pedestrians to cross safely.” It then continues with garbled and unrelated material. |

These are examples from the artifact, not proof that every response is correct. The notebook’s judge reasoning similarly notes that fine-tuned answers sometimes contain a relevant opening but degrade into unrelated text, repetitions, or stray tokens.

## 12. Training and Evaluation Curves

The notebook plots training loss and validation loss from the saved trainer history. Training loss trends downward overall, from approximately 0.82 at step 10 to approximately 0.72 at step 165, with fluctuations. Validation loss reaches its lowest recorded value around step 30 (about 1.36), then generally rises, reaching about 1.61 at step 165. This divergence is consistent with overfitting on a very small training set; it does not support a claim that validation performance steadily improved throughout training.

**Figure 2. Training and evaluation loss during SFT fine-tuning.**

> **Screenshot insertion placeholder:** The notebook contains a rendered training/validation curve, but no standalone loss-curve screenshot was present among the seven screenshot files reported by the student or in the workspace. Insert an exported screenshot of the notebook plot here. The reported `create sft trainer and train qlora  training.png` may document training setup/progress, but its contents were not available to verify that it shows the curve.

## 13. Qualitative Before / After Comparison

The saved examples indicate a clear but incomplete shift. The base model often produces unrelated or corrupted text. Fine-tuning sometimes yields a short, relevant opening for questions about mobile-phone use and pedestrian crossings, demonstrating domain alignment in the beginning of the response. In the examples examined, that improvement is undermined by extra corrupted tokens, repeated content, and unexpected continuation into unrelated questions. As a result, the output cannot be considered reliably usable without further decoding/training improvements and factual verification.

## 14. Demo / Inference

The notebook defines and launches a Gradio interface. A user enters a question, which is formatted as a Qwen user/assistant prompt and passed to the base model with the saved LoRA adapter. Generation is deterministic (`do_sample=False`), limited to 150 new tokens, and configured with a repetition penalty and stop-token IDs. The notebook output reports that the hosted Colab environment launched a temporary public Gradio URL. The URL was temporary and is not treated as a permanent deployment. The interface itself warns that answers can be incomplete or wrong and points users to official rules or local traffic police.

## 15. Engineering Design Principles

- **Domain data and validation:** The source data was traffic-law focused; the notebook inspected its real schema and parsed the combined text format instead of assuming separate fields.
- **Cleaning and reproducibility:** Whitespace was normalized, malformed/empty pairs were filtered, exact duplicates removed, and shuffling/splitting used seed 42. The three splits and formatted/tokenized dataset directories were saved.
- **Memory-aware fine-tuning:** 4-bit QLoRA and LoRA reduced the trainable set to about 1.18% of model parameters, fitting the available Colab T4 setup.
- **Pre-training checks:** The notebook printed library and GPU information, inspected records and sequence lengths, checked parsing and overlap, and printed trainable parameter counts before training. No separate short pilot-training run is verified.
- **Before/after evaluation:** Both models were compared on the same evaluation set, with loss/perplexity and a separate model-based qualitative judge. The limits of these measures are stated above.

## 16. Limitations

- The cleaned dataset has only 162 examples, with 129 for training and 17 for final evaluation.
- The evaluation split has one repeated question also found in training, and the evaluation sample is small.
- Validation loss worsened after its early minimum while training loss declined, indicating a likely generalization problem.
- Fine-tuned generations visibly contain corrupted, repetitive, and unrelated text after relevant openings; the model is not a dependable user-facing legal assistant in this state.
- Loss and perplexity measure language-model fit, not the factual correctness of legal guidance.
- GPT-OSS judging is model-based and cannot replace expert legal validation; no official legal review or large-scale benchmark is verified in the available artifacts.
- The epoch count is inconsistent between executed code/checkpoint evidence and the saved metrics summary, as documented in Section 8.

## 17. Final Outcome

A 162-example cleaned dataset was split and formatted; Qwen2.5-1.5B was configured for 4-bit QLoRA; the LoRA adapter was trained and saved; evaluation loss/perplexity and base-versus-fine-tuned comparisons were produced; an eight-question GPT-OSS judge comparison was saved; and a Gradio demo was launched in the notebook. The evidence supports completion of the technical pipeline, but not dependable legal answer quality.

## 18. Conclusion

The Task 2 fine-tuning pipeline was implemented and the adapted model achieved substantially lower evaluation loss and perplexity than the base model on the project’s small held-out set. The qualitative and judge evidence shows some improved traffic-domain relevance, but also persistent corrupted continuations and weak overall quality. The experiment therefore demonstrates successful parameter-efficient adaptation and measurable dataset-fit improvement, not validated legal reliability or broad generalization.

## Screenshot Evidence

The filenames below are the seven screenshot names reported by the student. The image files themselves were not present in the workspace during preparation, so the placeholders below should be replaced with actual image embeds when the files are added to the project folder.

### Screenshot 1 — Dataset Preparation and Train/Validation/Evaluation Split

**Candidate supplied files:** `load data.png`; `split data.png`.

**Figure 1. Dataset preprocessing, duplicate removal and train/validation/evaluation split.**

This evidence should show the source data being loaded, the cleaning/deduplication result, and the final counts of 129 training, 16 validation, and 17 evaluation examples. The available notebook output records a reduction from 200 raw rows to 162 unique question-answer pairs, with 38 exact duplicates removed. The screenshot files were not available to inspect, so their contents should be checked when inserted.

### Screenshot 2 — Training and Evaluation Loss Curves

**Candidate supplied file:** `create sft trainer and train qlora  training.png` (filename suggests training setup, but image contents were unavailable). The notebook contains the actual rendered training/validation loss plot; export that plot if this screenshot does not show it.

**Figure 2. Training and evaluation loss during SFT fine-tuning.**

The notebook curve shows training loss declining overall while validation loss bottoms out early and then rises toward the end of the five-epoch run. This divergence is consistent with overfitting and is a reason to interpret the low held-out text loss cautiously. Confirm that the inserted image actually shows the curve before retaining this caption.

### Screenshot 3 — Final Evaluation: Base vs Fine-Tuned Model

**No matching final-results screenshot was among the seven filenames reported.** `conf_sfttrainer.png`, `add lora adaptor.png`, `model set up.png`, and `load qwen tokenizwer.png` are other supplied filenames, but their contents could not be inspected and they should not be used as final evaluation evidence unless they show the requested results.

**Figure 3. Final evaluation comparison between the base and fine-tuned models.**

The recorded final summary reports base loss/perplexity of 4.5090/90.83 and fine-tuned loss/perplexity of 1.2387/3.45. It also reports eight judge comparisons: seven fine-tuned selections and one tie, with mean scores of 1.12 and 4.00 respectively. Insert a screenshot showing those results from the notebook before submission; the numerical summary here is sourced from the saved project artifacts.

## Source Artifacts Used

- `Road_Safety_QLoRA_Qwen2_5_1_5B.ipynb` — notebook code, embedded outputs, configuration, training/evaluation summaries, sample generations, plots and demo launch output.
- `road_safety_qwen2.5_1.5b_qlora/evaluation_metrics.json` — saved split and base/fine-tuned evaluation metrics; its epoch field conflicts with executed training evidence.
- `road_safety_qwen2.5_1.5b_qlora/model_comparison.csv` — saved qualitative base/fine-tuned examples.
- `road_safety_qwen2.5_1.5b_qlora/model_evaluation.csv` — eight model-judge results and reasoning.
- `road_safety_qwen2.5_1.5b_qlora/final_model/adapter_config.json` — saved base model and LoRA configuration.
- `road_safety_qwen2.5_1.5b_qlora/final_model/adapter_model.safetensors` — saved LoRA adapter weights.
- `road_safety_qwen2.5_1.5b_qlora/final_model/tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` — saved tokenizer/chat artifacts.
- `road_safety_qwen2.5_1.5b_qlora/checkpoints/checkpoint-165/trainer_state.json` — executed training history and final epoch/step evidence.
- `road_safety_qwen2.5_1.5b_qlora/checkpoints/checkpoint-165/adapter_config.json` — checkpoint LoRA configuration.
- `road_safety_qwen2.5_1.5b_qlora/checkpoints/README.md` and `road_safety_qwen2.5_1.5b_qlora/final_model/README.md` — model-card metadata; incomplete template fields were not treated as experimental evidence.
- User-reported screenshot filenames: `add lora adaptor.png`, `conf_sfttrainer.png`, `create sft trainer and train qlora  training.png`, `load data.png`, `load qwen tokenizwer.png`, `model set up.png`, and `split data.png`. These images were not present in the workspace and were not inspected.
