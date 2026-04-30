# Task 1 — Data Preprocessing Report (Part 1)

## English

### Overview

This report summarizes **Task 1**: building a cleaned SVG corpus, training a BPE tokenizer, splitting data by file, and recording dataset statistics. Processing was implemented in `scripts/task1/preprocess_dataset.py` with cleaning in `svg_scaling/cleaning.py`. Artifacts are stored under `data/processed/`; Task 1 figures and example renders are under `outputs/task1/`.

### Cleaned corpus

- **What it is:** Each Hugging Face row is one SVG **file** (keyed by dataset id and `Filename`). Raw SVG strings are passed through `clean_svg()`: XML parse with comment stripping, removal of `<metadata>`, `<title>`, and `<desc>`, whitespace normalization, float rounding (1 decimal), and re-validation as well-formed XML. Sequences outside character length bounds are dropped before splitting.
- **Outputs:** `data/processed/train.jsonl`, `val.jsonl`, and `test.jsonl`. Each line is a JSON object with cleaned `svg`, `num_tokens` (after BPE), `source_dataset`, and `filename`.
- **Scale after cleaning:** **364,130** files retained after cleaning (`kept_files_after_clean` in `stats.json`).

### Tokenizer

- **Library:** **SentencePiece** BPE (`sentencepiece`), not Hugging Face `tokenizers`.
- **Training:** The model is trained **only on the training split** text (one SVG per line, newlines collapsed to spaces), then applied to all splits.
- **Vocabulary size:** **4096** (within the suggested **1k–8k** range).
- **Artifacts:** `data/processed/spm.model`, `data/processed/spm.vocab`.
- **Length filter:** Sequences with more than **2048** BPE tokens are dropped (`max_token_len`).

#### Vocabulary size rationale (4096)

We set the SentencePiece vocabulary to **4096** because it sits in the middle of the **1K–8K** course range and offers a practical trade-off for SVG text: large enough to represent frequent tags, attributes, and numeric fragments as compact subwords, yet small enough to keep sequence lengths and embedding memory reasonable for modest decoder-only LMs. Very small vocabs increase sequence length and rare-token noise; very large vocabs add parameters with diminishing returns for this preprocessing stage, so **4096** is a standard default choice and matches our pipeline configuration.

#### Optional steps not applied in this run

- **Canonical attribute ordering** was **not** implemented (the assignment marks it optional). Sorting attributes lexicographically on every element would further stabilize the text form but adds implementation complexity (namespaces, duplicate handling) with limited benefit for our BPE objective; we rely on `lxml` serialization instead.
- **Full Cairo rasterization filtering** was **not** used during preprocessing (`--render-check` off). We instead **validated CairoSVG on a random sample** of `train.jsonl` to confirm the toolchain. A full Cairo gate would drop any SVG Cairo cannot rasterize and would require **re-running the entire preprocessing pipeline** (cleaning counts, splits, BPE, `stats.json`, and figures). To run it on Windows, install GTK/Cairo, ensure Python can load `libcairo-2.dll` (see `README.md`), activate the same venv, then from the repo root run the same `preprocess_dataset.py` command as for the merged corpus but add **`--render-check`**. Expect **much longer** runtime; if train tokens fall below **~100M**, increase `--fonts-subsample` (or add data) and re-run until the target is met.

### Train / validation / test split

- **Ratios:** **98% / 1% / 1%** (train / val / test).
- **Granularity:** **By file** — the list of cleaned file records is shuffled with a fixed seed (**42**), then partitioned. There is **no** splitting inside a single SVG sequence.
- **Sources merged:** `starvector/svg-icons-simple`, `starvector/svg-emoji-simple`, and a **subsample** of `starvector/svg-fonts-simple` (train split, capped row count) so that total training tokens meet the course target (≥ ~100M BPE tokens).

### Dataset statistics (summary)

| Item | Value |
|------|--------|
| Vocabulary size | **4096** |
| Train tokens | **102,899,186** |
| Validation tokens | **1,049,271** |
| Test tokens | **1,042,877** |

**Sequence length distribution** is summarized for the **training set** after tokenization (BPE length = `num_tokens`):

- **Histogram:** Stored in `data/processed/stats.json` under `train_length_histogram` (`bin_edges`, `counts`, and `quantiles`).
- **Quantiles (train):** min **59**; 25% **154**; median **203**; 75% **306**; 90% **571**; 99% **1347**; max **2048** (sequences at the cap are allowed up to 2048 tokens before filtering).

**Sample counts before / after filtering:**

- **After cleaning (before train/val/test split):** **364,130** files in one pool.
- **After 98/1/1 split, before the 2048-token cap:** **356,847** train + **3,641** val + **3,642** test = **364,130** (each split is encoded with the trained BPE; counts here include rows that will still be dropped if `num_tokens` > 2048).
- **After BPE + ≤2048 token filter (`files_after_token_filter`):** **356,748** train | **3,638** val | **3,641** test.
- **Dropped by token cap (>2048 BPE tokens):** **99** train + **3** val + **1** test = **103** total.

Equivalently: **364,130 − 103 = 364,027** files remain in the three jsonl files combined.

### Figures and example visuals

- **Statistics figure:** Sequence-length distribution is plotted from `stats.json` as **`outputs/task1/train_length_histogram.png`** (and **`train_length_histogram.pdf`** if generated) for the report.
- **Example SVGs by complexity:** **`outputs/task1/gallery.html`** embeds inline SVGs at approximate quantiles **0.10 / 0.50 / 0.90** of `num_tokens` on the training set, with companion `.svg` files in the same folder. These illustrate short-, medium-, and long-BPE examples for documentation.

### Reproducibility

- Random seed for splitting: **42** (`stats.json` → `seed`).
- Full configuration is recorded in `data/processed/stats.json`.


