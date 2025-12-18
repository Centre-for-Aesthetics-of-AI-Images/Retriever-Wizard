# 🧙‍♂️ Retriever Wizard

Retriever Wizard is a small web app (Streamlit) for exploring a collection of images by *visual similarity*, based on one image. This tool was built as part of an ongoing PhD project (IKK, Aarhus University).

You can use it to:
- Pick a “query” image and find other images based on computational similarity
- Browse results in a reading-friendly list (stacked view)
- Add your own categories/labels as an overlay (without changing your original metadata)
- Make a simple 2D “map” of the collection for exploratory browsing or communication

## Glossary
- **Metadata**: your spreadsheet about the images (title, date, author, reference, etc.).
- **Embeddings**: a table of numbers (one row per image) produced by a vision model; it lets the tool compare images by visual similarity.
- **Nearest neighbors**: the most visually similar images to your chosen query image.
- **Overlay / annotation**: your added categories/labels saved separately, so your source data remains untouched.

## Included example dataset
This repo includes an `examples/` test set (1k images) from a larger collection of educational wall charts.
- Ownership: Danish School of Education (DPU), Aarhus University.
- Source: Digitized by The Royal Danish Library; the selected example images are free of copyright (source hyperlink included in `examples/metadata.csv`).
- Research context: Processed during an ongoing PhD project; embeddings produced with `google/siglip2-giant-opt-patch16-384`.

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

## Install (Windows)
```
git clone https://github.com/Centre-for-Aesthetics-of-AI-Images/Retriever-Wizard.git
cd retriever-wizard

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
\.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run
```
streamlit run ReWiz.py
```

## Using the app
- **Step 1–3**: point the app at your metadata CSV, embeddings CSV, and image folder(s).
- **Step 4–5**: run a consistency check and build/load a file index (so thumbnails load quickly).
- **Step 6**: choose a query image, optionally filter by metadata, and find similar images.
- **Step 8**: add annotation fields (your categories) and apply them as an overlay.
- **Step 9**: allows a visual projection of the distance between the query image and other images.

If you’re using  known and already verified data, you can skip setup checks by clicking **Auto-validate Steps 1–5 and go to Step 6** (2-clicks needed).

## Notes
- The app stores a small settings checkpoint in `.retriever_wizard/settings.json` (next to `ReWiz.py`).
- Annotation overlays are written under your chosen `output_dir` in an `_overlay/` folder and never overwrite your input CSVs, but can later be used as potential metadata.

### Disclaimer
Parts of this project were drafted or refactored with the assistance of large language models.
