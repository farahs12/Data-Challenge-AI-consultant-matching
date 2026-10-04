# AI Consultant Matching Platform

Matches consultants to a mission/job description (PDF) by scoring skills fit, mission similarity, language requirements, and availability.

## How it works

1. A terms-of-reference PDF (in `documents/`) is read and sent to a local LLM (via [Ollama](https://ollama.com)) to extract structured requirements (job role, required technologies, languages, dates, etc.) — see `helpers/tools_extraction.py`.
2. Each required skill is embedded with a fine-tuned sentence-transformer and compared against pre-computed consultant skill embeddings (`skills/skills_encoded_hierarchy.npz`) — see `helpers/SkillsEmbeddings.py`.
3. Mission text similarity, language fit, and availability penalties are computed and combined into a final ranked list of consultants.

Two ways to run the pipeline:
- CLI: `mainfile/main.py`
- Web UI: `app.py` (Streamlit)

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally, with the model pulled:
  ```bash
  ollama pull gemma3:12b
  ```
- Internet access on first run (to download the base `BAAI/bge-m3` model and the `paraphrase-multilingual-mpnet-base-v2` mission embedding model from Hugging Face).

## Setup

```bash
python3 -m venv env
source env/bin/activate          # on Windows: env\Scripts\activate
pip install -r requirements.txt
```

> Note: a committed `env/` virtualenv should **not** be relied on — rebuild it locally with the commands above.

### Fine-tuned skill embedding model

`skills_finetuned_hierarchy/` and `skills_finetuned_roles/` (the fine-tuned embedding models) and their encoded `.npz` files are **not tracked in git** (see `.gitignore` — model weight files and `skills_finetuned_*/` are excluded). Before running the pipeline for the first time, regenerate them:

```bash
python notebooks/finetune_skill_hierarchy.py        # trains and saves skills_finetuned_hierarchy/
python notebooks/reencode_skills_with_hierarchy.py  # re-encodes skills/skills_encoded_hierarchy.npz to match
```

This trains on CPU from `skills/skillscleaned.csv` and takes a few minutes. Run both scripts in this order — the second step re-encodes consultant skills so they're in the same embedding space as the freshly trained model (re-running `finetune_skill_hierarchy.py` produces slightly different weights each time, so an out-of-date `.npz` won't match).

## Running

### CLI

```bash
python mainfile/main.py <name>
```

`<name>` can be a bare filename (matched against `documents/<name>.pdf`), a filename with extension, or a relative/absolute path to any PDF. Example:

```bash
python mainfile/main.py Scrum   # reads documents/Scrum.pdf
```

Prints the top 5 matched consultants.

### Web app

```bash
streamlit run app.py
```

Upload a PDF in the UI to extract requirements and browse ranked, filterable consultant matches.

## Configuration

All matching behavior is controlled by [`config/search_config.yaml`](config/search_config.yaml):

- `model`: which Ollama model extracts requirements from the PDF
- `embedding_model` / `skills_embedding_file`: which skill embedding model/encoded consultant data to use
- `skills`: matching thresholds and weighting
- `mission_context`, `languages`, `disponibility`: weighting and penalty settings for mission similarity, language fit, and availability

## Project structure

```
app.py                  Streamlit app
mainfile/main.py        CLI entry point
helpers/                Skill embedding + LLM extraction helpers
src/                    Mission/language/schedule scoring
notebooks/              Model fine-tuning and data exploration scripts
config/                 Matching configuration
data/, skills/          Consultant and skill datasets
documents/              Sample terms-of-reference PDFs
```
