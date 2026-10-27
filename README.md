# Road Safety QLoRA

A small domain-adaptation experiment that fine-tunes **Qwen2.5-1.5B** to answer questions about Indian traffic rules and road safety using **Supervised Fine-Tuning (SFT) with 4-bit QLoRA**.

> This is an educational prototype, not legal advice. Verify traffic-law information with official sources or local traffic authorities.

## Project Contents

- `Road_Safety_QLoRA_Qwen2_5_1_5B.ipynb` — end-to-end Google Colab notebook for data preparation, training, evaluation, plots, and Gradio demo.
- `road_safety_qwen2.5_1.5b_qlora/final_model/` — saved LoRA adapter and tokenizer files.
- `road_safety_qwen2.5_1.5b_qlora/checkpoints/` — saved training checkpoints and trainer state.
- `road_safety_qwen2.5_1.5b_qlora/evaluation_metrics.json` — saved evaluation metrics and split sizes.
- `road_safety_qwen2.5_1.5b_qlora/model_comparison.csv` — example base and fine-tuned generations.
- `road_safety_qwen2.5_1.5b_qlora/model_evaluation.csv` — GPT-OSS judge comparisons.
- `LLM_Engineering_Submission_Report.md` — detailed project report.

## Experiment

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B` |
| Dataset | `Pravincoder/Indian_traffic_law_QA` |
| Raw / cleaned examples | 200 / 162 |
| Train / validation / evaluation | 129 / 16 / 17 |
| Fine-tuning | SFT with LoRA adapters and 4-bit NF4 quantization |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Training hardware recorded in notebook | Google Colab Tesla T4 (14.56 GB measured device memory) |
| Demo | Gradio interface defined in the notebook |

### Evaluation Summary

| Metric | Base model | Fine-tuned model |
|---|---:|---:|
| Evaluation loss | 4.5090 | 1.2387 |
| Perplexity | 90.83 | 3.45 |
| Mean GPT-OSS judge score | 1.12 | 4.00 |

The judge compared eight question-answer pairs: it selected the fine-tuned model seven times and recorded one tie. Loss and perplexity show fit to the small formatted evaluation set; they do not establish factual or legal correctness. Sample generations contain garbled and repetitive continuations, including after relevant answer openings.

## Run the Notebook

1. Open `Road_Safety_QLoRA_Qwen2_5_1_5B.ipynb` in Google Colab.
2. Select a GPU runtime. The recorded experiment used a Tesla T4.
3. Run the notebook cells in order. It installs the training libraries, loads the Hugging Face dataset, prepares splits, trains the adapter, evaluates the models, and launches the Gradio demo.
4. The optional GPT-OSS judge step requires a Groq API key. Provide it through Colab Secrets as `GROQ_API_KEY`; do not commit keys to GitHub.

The notebook saves its generated artifacts to Google Drive under `road_safety_qwen2.5_1.5b_qlora`.

## Training Configuration Note

The executed `SFTConfig`, training output, and `checkpoint-165/trainer_state.json` indicate **5 epochs** and 165 steps. Some notebook Markdown, `evaluation_metrics.json`, and the final-results summary say **1 epoch**. Treat five epochs as the executed run; the one-epoch value is inconsistent metadata. Validation loss reached its lowest recorded point early and rose later in training, suggesting overfitting on this small dataset.
