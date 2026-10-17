# QUIRE — Local-first PDF workspace

## What is included
- Smart PDF analysis: extractive summary, keywords, lightweight category matching
- Compare two PDFs using TF-IDF cosine similarity
- Merge PDFs
- Split a page range
- Rotate all, odd, or even pages
- Optimize/rewrite a PDF with PyMuPDF compression options
- Convert images to PDF
- Convert DOCX paragraph text to PDF
- Extract text and PDF metadata
- Responsive layout and an in-app Light/Dark theme switch
- Separate theme logos when supplied

## Logo files
Place your two original, high-resolution logo images next to `app.py` and name them exactly:
- `Quire_light.png`
- `Quire_dark.png`

PNG is recommended. The original files are embedded as data URIs, not resized/re-encoded by Python. If either file is absent, the app uses `Quire1.png` if present, or a simple text mark.

## Run locally (no cloud/server account required)
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   `pip install -r requirements.txt`
5. Start the app:
   `streamlit run app.py`

It opens in your browser at a local address (usually http://localhost:8501). This uses a local-only web server process on your computer; it does not require a remote server or upload files to Streamlit Cloud. Do not expose the port to your network if you want the app to remain accessible only on this computer.

## Important limitations
- This is a local-first Streamlit app, not a compiled standalone executable. Python and dependencies must be installed.
- DOCX conversion preserves paragraph text only; it does not reproduce Word tables, images, headers, footers, or complex page layout.
- Text similarity is a statistical overlap estimate, not a plagiarism verdict.
- The ML analysis is lightweight TF-IDF-based analysis; it is not a cloud LLM and does not require internet after dependencies are installed.
- Scanned PDFs require OCR, which is not included.
- PDF optimization does not guarantee smaller files; already-compressed images may dominate the size.
- Running the app on Streamlit Cloud means files are processed on that remote host. For on-device processing, run it locally using the steps above.
