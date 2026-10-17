import io
import re
import base64
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# QUIRE — local-first PDF workspace
# All document processing is performed by this Python process on the machine running it.
st.set_page_config(page_title="QUIRE · PDF workspace", page_icon="Quire_favicon.png", layout="wide",
                   initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent

def read_logo(candidates):
    for name in candidates:
        p = APP_DIR / name
        if p.exists():
            try:
                return base64.b64encode(p.read_bytes()).decode("ascii"), p.suffix.lower().lstrip(".")
            except OSError:
                pass
    return None, None

light_logo, light_ext = read_logo(["Quire_light.png", "Quire1.png", "quire_light.png"])
dark_logo, dark_ext = read_logo(["Quire_dark.png", "Quire1.png", "quire_dark.png"])

if "quire_theme" not in st.session_state:
    st.session_state.quire_theme = "Dark"

def apply_theme():
    dark = st.session_state.quire_theme == "Dark"
    if dark:
        bg, panel, text, muted, border, accent = "#0c1018", "#141b27", "#edf2fb", "#9ba9bd", "#283449", "#6ee7c0"
        field, soft = "#101724", "#172235"
    else:
        bg, panel, text, muted, border, accent = "#f5f7fb", "#ffffff", "#172033", "#69758a", "#e1e6ef", "#087f6e"
        field, soft = "#ffffff", "#edf7f4"
    logo_css = ""
    if light_logo and dark_logo:
        mime_l = "image/svg+xml" if light_ext == "svg" else "image/png"
        mime_d = "image/svg+xml" if dark_ext == "svg" else "image/png"
        logo_css = f"""
        .quire-logo-light {{ display: {'none' if dark else 'block'}; }}
        .quire-logo-dark {{ display: {'block' if dark else 'none'}; }}
        """
    else:
        mime_l = "image/png"; mime_d = "image/png"
    st.markdown(f"""
    <style>
    :root {{ color-scheme: {'dark' if dark else 'light'}; }}
    [data-testid="stAppViewContainer"] {{ background: {bg}; color: {text}; }}
    [data-testid="stHeader"] {{ background: {bg}; }}
    [data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
    [data-testid="stSidebar"] * {{ color: {text}; }}
    [data-testid="stMainBlockContainer"] {{ max-width: 1440px; padding-top: 2.5rem; padding-bottom: 3rem; }}
    h1,h2,h3,h4,p,li,label {{ color: {text}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {muted} !important; }}
    [data-testid="stMarkdownContainer"] p {{ line-height: 1.65; }}
    [data-testid="stFileUploader"] section {{ background: {field}; border: 1px dashed {border}; border-radius: 16px; }}
    [data-testid="stFileUploader"] section:hover {{ border-color: {accent}; }}
    
    /* Fix file upload button in both themes */
    [data-testid="stFileUploader"] button {{
        background: {panel} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        opacity: 1 !important;
    }}

    [data-testid="stFileUploader"] button p,
    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button svg {{
        color: {text} !important;
        fill: {text} !important;
    }}

    [data-testid="stFileUploader"] button:hover {{
        background: {soft} !important;
        color: {text} !important;
        border-color: {accent} !important;
    }}

    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
       background: {field}; color: {text}; border-color: {border}; border-radius: 10px;
    }}
    [data-testid="stMetric"] {{ background: {panel}; border: 1px solid {border}; padding: 18px 20px; border-radius: 16px; }}
    [data-testid="stMetricLabel"] {{ color: {muted}; }}
    [data-testid="stMetricValue"] {{ color: {text}; }}
    .stButton > button, .stDownloadButton > button {{ border-radius: 11px; min-height: 2.65rem; font-weight: 600; border: 1px solid {border}; }}
    .stButton > button[kind="primary"] {{ background: #8ce2c1 !important; color: #08221d !important; border: 0; }}
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {{ color: #08221d !important; }}
    .stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {accent}; transform: translateY(-1px); }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 1px solid {border}; }}
    [data-testid="stTabs"] button[role="tab"] {{ border-radius: 10px 10px 0 0; padding: .7rem 1rem; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color: {accent}; border-bottom-color: {accent}; }}
    div[data-testid="stExpander"] {{ border: 1px solid {border}; border-radius: 13px; overflow: hidden; }}
    hr {{ border-color: {border}; }}
    .quire-hero {{ background: linear-gradient(125deg, {panel}, {soft}); border: 1px solid {border}; border-radius: 22px; padding: 24px 28px; margin-bottom: 22px; }}
    .quire-eyebrow {{ color: {accent}; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 800; }}
    .quire-title {{ font-size: clamp(1.7rem, 3vw, 2.65rem); font-weight: 850; letter-spacing: -.045em; line-height: 1.1; margin: 8px 0 9px; color: {text}; }}
    .quire-subtitle {{ color: {muted}; font-size: .98rem; margin: 0; }}
    .quire-card {{ background: {panel}; border: 1px solid {border}; border-radius: 16px; padding: 18px; height: 100%; }}
    .quire-card-title {{ font-size: 1rem; font-weight: 750; margin-bottom: 6px; }}
    .quire-card-desc {{ color: {muted}; font-size: .88rem; line-height: 1.5; }}
    .quire-pill {{ display:inline-block; padding: 4px 9px; border-radius: 999px; background: {soft}; color: {accent}; font-size: .75rem; font-weight: 750; }}
    .quire-logo {{ width: 84px; height: 84px; object-fit: contain; }}
    .quire-brand {{ display:flex; gap:12px; align-items:center; margin: 2px 0 20px; }}
    .quire-brand-name {{ font-size:1.2rem; font-weight:850; letter-spacing:-.03em; color:{text}; }}
    .quire-brand-caption {{ font-size:.78rem; color:{muted}; }}
    {logo_css}
    
    /* Sidebar expand/collapse arrow — visible in light mode */
    [data-testid="stSidebarCollapsedControl"] span[data-testid="stIconMaterial"],
    [data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"] {{
        color: {'#ffffff' if dark else '#000000'} !important;
    }}

    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stExpandSidebarButton"] svg {{
        color: {'#ffffff' if dark else '#000000'} !important;
        fill: {'#ffffff' if dark else '#000000'} !important;
    }}
    
    
    /* QUIRE navigation: color only the circle wrapper */
    [data-testid="stRadioGroup"] label[data-testid="stRadioOption"]:not([data-selected="true"]) > div > div:first-child {{
        background-color: {"" if dark else "#EFF7F4"} !important;
        border-color: {"" if dark else "#BDD3CB"} !important;
    }}

    /* Selected navigation circle */
    [data-testid="stRadioGroup"] label[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {{
        background-color: {'#F15B55'} !important;
        border-color: {'#F15B55'} !important;
    }}

    /* Keep tool labels free of colored backgrounds */
    [data-testid="stRadioGroup"] label[data-testid="stRadioOption"],
    [data-testid="stRadioGroup"] label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"],
    [data-testid="stRadioGroup"] label[data-testid="stRadioOption"] p {{
        background: transparent !important;
    }}
    
    
    /* Appearance toggle — inactive button */
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="false"] {{
        background-color: {'rgba(79, 129, 119, 0.12)' if dark else 'rgba(79, 129, 119, 0.12)'} !important;
        color: {'#E8EEF5' if dark else '#4F8177'} !important;
        border-color: rgba(79, 129, 119, 0.45) !important;
    }}

    /* Appearance toggle — active button */
    [data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {{
        background-color: {'#30252D' if dark else '#FCEDEC'} !important;
        color: {'#F4F5F7' if dark else '#202536'} !important;
        border-color: {'#F15B55'} !important;
    }}


   @media (max-width: 700px) {{
      [data-testid="stMainBlockContainer"] {{ padding: 1rem .8rem 2rem; }}
      .quire-hero {{ padding: 19px 17px; border-radius: 17px; }}
      .quire-logo {{ width: 58px; height: 58px; }}
      .quire-brand-name {{ font-size: 1.05rem; }}
      [data-testid="stTabs"] [data-baseweb="tab-list"] {{ overflow-x:auto; flex-wrap:nowrap; }}
      [data-testid="stTabs"] button[role="tab"] {{ white-space:nowrap; padding:.6rem .7rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

def image_data_uri(b64, ext):
    if not b64:
        return ""
    mime = "image/svg+xml" if ext == "svg" else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/png")
    return f"data:{mime};base64,{b64}"

def render_brand():
    if light_logo and dark_logo:
        light_uri = image_data_uri(light_logo, light_ext)
        dark_uri = image_data_uri(dark_logo, dark_ext)
        light_html = f'<img class="quire-logo quire-logo-light" src="{light_uri}" alt="QUIRE logo">'
        dark_html = f'<img class="quire-logo quire-logo-dark" src="{dark_uri}" alt="QUIRE logo">'
        logo_html = light_html + dark_html
    elif light_logo:
        logo_html = f'<img class="quire-logo" src="{image_data_uri(light_logo, light_ext)}" alt="QUIRE logo">'
    else:
        logo_html = '<div class="quire-logo" style="display:grid;place-items:center;font-size:2.5rem">◈</div>'
    st.markdown(f"""
    <div class="quire-brand">{logo_html}
      <div><div class="quire-brand-name">QUIRE <span class="quire-pill">PDF WORKSPACE</span></div>
      <div class="quire-brand-caption">Private by design · Process files locally</div></div>
    </div>
    """, unsafe_allow_html=True)

def pdf_bytes_from_doc(doc):
    out = io.BytesIO()
    doc.save(out, garbage=4, deflate=True, clean=True)
    #doc.close()
    return out.getvalue()

def extract_text_from_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc).strip(), len(doc)

def merge_pdfs(pdf_list):
    merged = fitz.open()
    for b in pdf_list:
        with fitz.open(stream=b, filetype="pdf") as src:
            merged.insert_pdf(src)
    return pdf_bytes_from_doc(merged)

def split_pdf(pdf_bytes, start_page, end_page):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as src:
        out = fitz.open()
        out.insert_pdf(src, from_page=start_page - 1, to_page=end_page - 1)
        return pdf_bytes_from_doc(out)

def rotate_pdf(pdf_bytes, rotation, pages_text):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for idx, page in enumerate(doc):
            if pages_text == "All pages" or (pages_text == "Odd pages" and idx % 2 == 0) or (pages_text == "Even pages" and idx % 2 == 1):
                page.set_rotation((page.rotation + rotation) % 360)
        return pdf_bytes_from_doc(doc)

def compress_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return pdf_bytes_from_doc(doc)

def images_to_pdf(files):
    images = []
    for f in files:
        im = Image.open(io.BytesIO(f.getvalue())).convert("RGB")
        images.append(im)
    if not images:
        return None
    out = io.BytesIO()
    images[0].save(out, format="PDF", save_all=True, append_images=images[1:], resolution=150)
    return out.getvalue()

def docx_to_pdf(docx_bytes):
    word = Document(io.BytesIO(docx_bytes))
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    margin, y, line_height = 48, 52, 15
    max_y = 790
    for para in word.paragraphs:
        text = para.text.strip()
        if not text:
            y += 8
            continue
        # Basic text wrapping; this is a text-focused conversion, not a full Word renderer.
        words = text.split()
        lines, current = [], ""
        for word_part in words:
            candidate = (current + " " + word_part).strip()
            if fitz.get_text_length(candidate, fontsize=10) > 495 and current:
                lines.append(current); current = word_part
            else:
                current = candidate
        if current:
            lines.append(current)
        for line in lines:
            if y > max_y:
                page = pdf.new_page(width=595, height=842); y = 52
            page.insert_text((margin, y), line, fontsize=10)
            y += line_height
        y += 5
    return pdf_bytes_from_doc(pdf)

def extract_keywords(text, top_n=8):
    if len(text.split()) < 5:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=150, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([text])
    terms = np.array(vectorizer.get_feature_names_out())
    scores = matrix.toarray()[0]
    ids = scores.argsort()[-top_n:][::-1]
    return [(str(terms[i]), round(float(scores[i]), 3)) for i in ids if scores[i] > 0]

def summarize_text(text, num_sentences=4):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:
        return "Text is too short or repetitive to summarize."
    scores = matrix.sum(axis=1).A1
    indices = sorted(scores.argsort()[-num_sentences:])
    return " ".join(sentences[i] for i in indices)

def classify_document(text):
    categories = {
        "Technology & Computer Science": "programming software python algorithm database artificial intelligence machine learning network hardware data web cloud",
        "Business, Finance & Economics": "market revenue sales investment stock growth finance accounting profit banking quarterly shares economy management",
        "Legal & Regulatory": "law contract agreement jurisdiction court terms clause party statutory compliant regulation compliance liability",
        "Academic & Research": "methodology experiment analysis hypothesis citation results literature study paper journal evaluation participants",
        "Medical & Healthcare": "patient diagnosis clinical treatment disease medical surgery symptoms therapy health pharmaceutical hospital"
    }
    names = list(categories)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([text] + list(categories.values()))
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    ranked = sorted([(name, round(float(score) * 100, 2)) for name, score in zip(names, scores)], key=lambda x: x[1], reverse=True)
    return ranked[0][0], ranked[0][1], ranked

def compute_pdf_similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    matrix = TfidfVectorizer(stop_words="english").fit_transform([a, b])
    return round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100, 2)

def download(label, data, filename, mime="application/pdf", key=None):
    st.download_button(label, data=data, file_name=filename, mime=mime, use_container_width=True, key=key)

# Sidebar: navigation + theme switch (app-controlled so logo switching is deterministic)
with st.sidebar:
    render_brand()
    st.markdown("### Workspace")
    st.caption("Choose a tool")
    tool = st.radio("Tools", [
        "Home", "Smart PDF analysis", "Compare PDFs", "Merge PDFs", "Split PDF",
        "Rotate pages", "Compress PDF", "Images to PDF", "Word to PDF", "Text extractor"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("### Appearance")
    st.segmented_control("Theme", options=["Light", "Dark"], key="quire_theme", label_visibility="collapsed")
    st.caption("Files are processed by this app instance. For offline use, run it on your own computer.")

# Re-apply colors after theme selection changes in the sidebar
apply_theme()

def hero(title, subtitle, tag="LOCAL PDF TOOLS"):
    st.markdown(f"""
    <div class="quire-hero">
      <div class="quire-eyebrow">{tag}</div>
      <div class="quire-title">{title}</div>
      <p class="quire-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def upload_box(label, types, key, multiple=False, help_text=None):
    return st.file_uploader(label, type=types, key=key, accept_multiple_files=multiple, help=help_text)

if tool == "Home":
    hero("Your documents. Your device.", "A clean, local-first PDF toolkit for everyday document work.", "WELCOME TO QUIRE")
    c1, c2, c3 = st.columns(3)
    for col, title, desc, icon in [
        (c1, "Organize", "Merge, split, rotate and compress PDFs.", "01"),
        (c2, "Convert", "Turn images and Word text into PDFs.", "02"),
        (c3, "Understand", "Extract text, compare files and analyze content.", "03")
    ]:
        with col:
            st.markdown(f'<div class="quire-card"><div class="quire-pill">{icon} / TOOLSET</div><div class="quire-card-title" style="margin-top:12px">{title}</div><div class="quire-card-desc">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Quick actions")
    quick = st.columns(4)
    quick_labels = [("Merge PDFs", "Combine documents"), ("Split PDF", "Extract page ranges"), ("Compress PDF", "Optimize structure"), ("Text extractor", "Get searchable text")]
    for col, (name, desc) in zip(quick, quick_labels):
        with col:
            st.markdown(f'<div class="quire-card"><div class="quire-card-title">{name}</div><div class="quire-card-desc">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.info("Privacy note: when you run QUIRE locally, your documents stay on your computer. A hosted Streamlit deployment processes files on the hosting server instead.")
    st.subheader("Available tools")
    st.write("Use the sidebar to open a tool. The same PDF processing features are available on desktop and mobile layouts.")

elif tool == "Smart PDF analysis":
    hero("Smart PDF analysis", "Extractive summary, key phrases and a lightweight topic classification.")
    file = upload_box("Choose a PDF to analyze", ["pdf"], "ml_pdf")
    if file:
        raw = file.getvalue()
        try:
            text, pages = extract_text_from_pdf(raw)
            a, b, c = st.columns(3)
            a.metric("Pages", pages); b.metric("Words", len(text.split())); c.metric("Characters", len(text))
            if not text:
                st.warning("No selectable text found. This may be a scanned PDF; OCR is not included in this offline build.")
            elif st.button("Analyze document", type="primary", use_container_width=True):
                if len(text.split()) < 30:
                    st.warning("The document has fewer than 30 words, so analysis may be unreliable.")
                best, confidence, ranked = classify_document(text)
                left, right = st.columns([1, 1], gap="large")
                with left:
                    st.subheader("Predicted category")
                    st.markdown(f'<div class="quire-card"><div class="quire-pill">TOP MATCH</div><h3 style="margin-top:12px">{best}</h3><p>Similarity score: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
                    st.subheader("Key phrases")
                    for kw, score in extract_keywords(text):
                        st.markdown(f"- **{kw}** · `{score}`")
                    with st.expander("All category scores"):
                        st.dataframe(pd.DataFrame(ranked, columns=["Category", "Similarity (%)"]), use_container_width=True, hide_index=True)
                with right:
                    st.subheader("Executive summary")
                    st.write(summarize_text(text, 4))
                    st.download_button("Download extracted text", text, file_name=f"{Path(file.name).stem}_text.txt", mime="text/plain", use_container_width=True)
        except Exception as e:
            st.error(f"Could not analyze this PDF: {e}")

elif tool == "Compare PDFs":
    hero("Compare two PDFs", "Estimate textual overlap using TF-IDF cosine similarity. This is not a formal plagiarism verdict.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        f1 = upload_box("Base document · A", ["pdf"], "sim_a")
    with c2:
        f2 = upload_box("Comparison document · B", ["pdf"], "sim_b")
    if f1 and f2 and st.button("Compare documents", type="primary", use_container_width=True):
        try:
            t1, p1 = extract_text_from_pdf(f1.getvalue())
            t2, p2 = extract_text_from_pdf(f2.getvalue())
            score = compute_pdf_similarity(t1, t2)
            st.metric("Text similarity", f"{score:.2f}%")
            st.progress(min(int(score), 100))
            if score > 70: st.error("High textual similarity. Review the documents closely.")
            elif score > 35: st.warning("Moderate textual similarity. The documents share terms or subject matter.")
            else: st.success("Low textual similarity based on extracted text.")
            st.caption(f"Document A: {p1} pages · Document B: {p2} pages")
        except Exception as e:
            st.error(f"Could not compare these PDFs: {e}")

elif tool == "Merge PDFs":
    hero("Merge PDFs", "Combine multiple PDF files into one document in the order you upload them.")
    files = upload_box("Choose two or more PDFs", ["pdf"], "merge_files", multiple=True)
    if files:
        st.write("**Merge order**")
        for i, f in enumerate(files, 1):
            st.write(f"{i}. {f.name}")
        if len(files) < 2:
            st.info("Add at least two PDFs to merge.")
        elif st.button("Merge files", type="primary", use_container_width=True):
            try:
                out = merge_pdfs([f.getvalue() for f in files])
                st.success("PDFs merged successfully.")
                download("Download merged PDF", out, "quire_merged.pdf")
            except Exception as e:
                st.error(f"Could not merge files: {e}")

elif tool == "Split PDF":
    hero("Split a PDF", "Extract a continuous page range into a new PDF.")
    file = upload_box("Choose a PDF", ["pdf"], "split_file")
    if file:
        data = file.getvalue()
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                total = len(doc)
            st.metric("Total pages", total)
            c1, c2 = st.columns(2)
            with c1: start = st.number_input("First page", 1, total, 1, key="split_start")
            with c2: end = st.number_input("Last page", 1, total, min(total, 2), key="split_end")
            if st.button("Extract page range", type="primary", use_container_width=True):
                if start > end: st.error("The first page must not be after the last page.")
                else:
                    out = split_pdf(data, int(start), int(end))
                    st.success(f"Extracted pages {start}–{end}.")
                    download("Download extracted PDF", out, f"pages_{start}_to_{end}.pdf")
        except Exception as e:
            st.error(f"Could not read this PDF: {e}")

elif tool == "Rotate pages":
    hero("Rotate PDF pages", "Correct page orientation without exporting through another service.")
    file = upload_box("Choose a PDF", ["pdf"], "rotate_file")
    if file:
        data = file.getvalue()
        with fitz.open(stream=data, filetype="pdf") as doc: total = len(doc)
        c1, c2 = st.columns(2)
        with c1: angle = st.selectbox("Rotation", [90, 180, 270], format_func=lambda x: f"{x}° clockwise")
        with c2: pages = st.selectbox("Apply to", ["All pages", "Odd pages", "Even pages"])
        if st.button("Rotate PDF", type="primary", use_container_width=True):
            out = rotate_pdf(data, angle, pages)
            st.success("Page rotation applied.")
            download("Download rotated PDF", out, f"{Path(file.name).stem}_rotated.pdf")

elif tool == "Compress PDF":
    hero("Optimize a PDF", "Rewrite the PDF with garbage collection and stream compression. Results depend on the document.")
    file = upload_box("Choose a PDF", ["pdf"], "compress_file")
    if file and st.button("Optimize PDF", type="primary", use_container_width=True):
        try:
            original = file.getvalue()
            optimized = compress_pdf(original)
            before, after = len(original), len(optimized)
            c1, c2, c3 = st.columns(3)
            c1.metric("Original size", f"{before / 1024:.1f} KB")
            c2.metric("Optimized size", f"{after / 1024:.1f} KB")
            delta = (before-after)/before*100 if before else 0
            c3.metric("Size change", f"{delta:+.1f}%")
            if after >= before:
                st.info("This PDF was already optimized, or its streams do not compress further. The output may be the same size or slightly larger.")
            download("Download optimized PDF", optimized, f"{Path(file.name).stem}_optimized.pdf")
        except Exception as e:
            st.error(f"Could not optimize this PDF: {e}")

elif tool == "Images to PDF":
    hero("Images to PDF", "Combine PNG and JPEG images into a single PDF.")
    files = upload_box("Choose images", ["png", "jpg", "jpeg"], "images_pdf", multiple=True)
    if files:
        st.caption(f"{len(files)} image(s) selected. Upload order determines page order.")
        for f in files:
            st.write(f"• {f.name}")
        if st.button("Convert images", type="primary", use_container_width=True):
            try:
                out = images_to_pdf(files)
                if out:
                    st.success("Images converted.")
                    download("Download PDF", out, "quire_images.pdf")
            except Exception as e:
                st.error(f"Could not convert images: {e}")

elif tool == "Word to PDF":
    hero("Word to PDF", "Convert DOCX paragraph text into a clean PDF. Complex layouts, tables and images may not be preserved.")
    file = upload_box("Choose a DOCX document", ["docx"], "docx_pdf")
    if file and st.button("Convert to PDF", type="primary", use_container_width=True):
        try:
            out = docx_to_pdf(file.getvalue())
            st.success("DOCX text converted to PDF.")
            download("Download PDF", out, f"{Path(file.name).stem}.pdf")
        except Exception as e:
            st.error(f"Could not convert this Word document: {e}")

elif tool == "Text extractor":
    hero("Extract text & metadata", "Pull selectable text from a PDF and save it as a plain-text file.")
    file = upload_box("Choose a PDF", ["pdf"], "text_pdf")
    if file:
        try:
            raw = file.getvalue()
            text, pages = extract_text_from_pdf(raw)
            with fitz.open(stream=raw, filetype="pdf") as doc:
                metadata = doc.metadata or {}
            a, b, c = st.columns(3)
            a.metric("Pages", pages); b.metric("Characters", len(text)); c.metric("Words", len(text.split()))
            with st.expander("PDF metadata", expanded=False):
                st.json({k: v for k, v in metadata.items() if v})
            st.text_area("Extracted text", value=text, height=360)
            st.download_button("Download .txt", text, file_name=f"{Path(file.name).stem}.txt", mime="text/plain", use_container_width=True)
            if not text:
                st.warning("No selectable text found. Scanned pages need OCR, which is not included in this offline build.")
        except Exception as e:
            st.error(f"Could not extract text: {e}")

st.divider()
st.caption("QUIRE · Local-first PDF workspace · PDF operations and analysis run locally when you launch this app on your own computer.")
