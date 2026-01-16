# 🧙‍♂️ Retriever Wizard

Retriever Wizard is a small web app (Streamlit) for exploring a collection of images by *visual similarity*. This tool was built as part of an ongoing PhD project (IKK, Aarhus University).

You can use it to:
- Pick a “query” image and find other images based on computational similarity
- Browse results in a reading-friendly list (stacked view)
- Add your own scholarly categories/notes as an overlay (without changing your original metadata)
- Make a simple 2D “map” of the collection for exploratory browsing or communication

## Glossary
- **Metadata**: your spreadsheet about the images (title, date, author, reference, etc.).
- **Embeddings**: a table of numbers (one row per image) produced by a vision model; it lets the tool compare images by visual similarity.
- **Nearest neighbors**: the most visually similar images to your chosen query image.
- **Overlay / annotation**: your added categories/labels saved separately, so your source data remains untouched.

## Included example dataset
This repo ships an `examples/` test set (images + metadata + embeddings) from a larger collection of educational wall charts.
- Ownership: Danish School of Education (DPU), Aarhus University.
- Source: Digitized by The Royal Danish Library; source hyperlinks live in `examples/metadata.csv`.
- Research context: Embeddings produced with `google/siglip2-giant-opt-patch16-384`.

`ReWiz.py` defaults to using:
- `examples/metadata.csv`
- `examples/embeddings.csv`
- `examples/images/`

## What you need to use your own collection
Retriever Wizard needs three things:
1) A folder of image files
2) A metadata spreadsheet saved as CSV (your descriptive fields)
3) An embeddings CSV (the “visual features” numbers, one row per image)

## How your files should match
- The tool connects everything through **filenames**.
- In practice, this means your metadata CSV and embeddings CSV need a column called `filename`.
- If your metadata uses a different column name (like `path` or `full_path`), the app will try to derive filenames automatically.
- Filenames should be unique (the app matches by basename, case-insensitive).

## Install (Windows / PowerShell)
```powershell
git clone <YOUR_REPO_URL>
cd retriever-wizard

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
\.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run
```powershell
	.\.venv\Scripts\streamlit.exe run ReWiz.py

Windows convenience:
	.\start_rewiz.cmd
	# or
	powershell -ExecutionPolicy Bypass -File .\start_rewiz.ps1
```

## Changing default paths (without UI)
Edit [rewiz_default_paths.json](rewiz_default_paths.json) to change the default:
- `meta_path`
- `embed_path`
- `images_root`
- `output_dir`

Relative paths are resolved relative to the repo folder.

## Using the app
- **Step 1–3**: point the app at your metadata CSV, embeddings CSV, and image folder(s).
- **Step 4–5**: run a consistency check and build/load a file index (so thumbnails load quickly).
- **Step 6**: choose a query mode, optionally filter by metadata, and find similar images.
- **Step 7**: review results in a reading-friendly stacked view.
- **Step 8**: add annotation fields (new columns) and apply them as an overlay.
- **Step 9**: allows a visual projection of the distance between the query image and other images.

If you’re using known data, you can skip setup checks by clicking **Auto-validate Steps 1–5 and go to Step 6**.

## Notes
- The app stores a small settings checkpoint in `.retriever_wizard/settings.json` (next to `ReWiz.py`).
- Annotation overlays are written under your chosen `output_dir` in an `_overlay/` folder and never overwrite your input CSVs.

## Step 6: Query modes
Retriever Wizard supports several query types:
- **Whole-image**: classic nearest-neighbor search. Uses your existing FAISS index (fast).
- **Text**: uses SigLIP2 (Transformers) to embed text into the same space as the image embeddings (requires your embeddings/FAISS to be built in SigLIP2 space).
- **Crop (light)**: uses SigLIP2 to embed only a cropped region as the query.
- **Crop (patch index)**: searches a patch-level index (optional, heavier setup).

Step 7 shows the “query” at the top based on your last search:
- Text shows the prompts used.
- Crop shows the crop preview.
- Whole-image shows the query image.

## Step 8: Overlay annotations (new columns)
Annotations are stored separately as an **overlay** and exported as merged CSVs (your source metadata is never overwritten).

Field types:
- **Category**: choose from a controlled vocabulary (optionally allow custom values).
- **Number**: numeric field (int/float).

Export styles:
- **Smart columns (recommended)**: exports a single column for single-choice fields; multi-choice fields export as a list-like string.
- **Wide / one-hot**: multi-choice fields become multiple 0/1 columns.
- **Binary**: exports presence/absence columns for selected category values.

## Troubleshooting
- **FAISS AVX512 vs AVX2 on Windows**: It’s normal to see FAISS try AVX512 and fall back to AVX2. As long as it ends with “Successfully loaded faiss with AVX2 support”, you’re fine.
- **Text / Crop (light) search dependencies**: These modes use Hugging Face Transformers. If you see warnings about missing tokenizers or “fast processor” requirements, ensure `torch`, `torchvision`, and `sentencepiece` are installed (they are included in `requirements.txt`).
- **Offline / restricted networks**: If model downloads time out, run once on a network that can download the SigLIP2 weights, or set `REWIZ_HF_LOCAL_ONLY=1` to force cache-only mode.

## Repo hygiene
- Cropped query images are stored under `_crops/` for convenience during a session and are ignored by git via `.gitignore`.

## Known gaps
- Navigation between steps can be smoother.

### Disclaimer
Parts of this project were drafted or refactored with the assistance of large language models.
