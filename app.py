import io
import re

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
st.set_page_config(page_title="PDF Table to Excel", layout="wide")
st.title("PDF/Image Table to Excel")

uploaded_file = st.file_uploader(
    "Upload PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"]
)


def pil_to_cv(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200):
    images = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page in pdf:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    pdf.close()
    return images


def detect_table_region(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Binary inverse threshold
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal and vertical lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    table_mask = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 50000 and w > 300 and h > 150:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def ocr_table_region(roi_bgr):
    text = pytesseract.image_to_string(roi_bgr, config="--psm 6")
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    return lines


def lines_to_dataframe(lines):
    # Basic splitter on 2+ spaces or tabs
    rows = []
    for line in lines:
        parts = re.split(r"\t+|\s{2,}", line)
        parts = [clean_text(p) for p in parts if clean_text(p)]
        if len(parts) >= 2:
            rows.append(parts)

    if not rows:
        return pd.DataFrame()

    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]

    df = pd.DataFrame(rows)

    # First row as header if it looks like one
    first_row = [str(x).strip() for x in df.iloc[0].tolist()]
    alpha_cells = sum(any(ch.isalpha() for ch in cell) for cell in first_row)

    if alpha_cells >= max(2, len(first_row) // 2):
        headers = []
        seen = {}
        for i, h in enumerate(first_row, start=1):
            h = h if h else f"Column_{i}"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 1
            headers.append(h)

        df = df.iloc[1:].reset_index(drop=True)
        df.columns = headers
    else:
        df.columns = [f"Column_{i+1}" for i in range(df.shape[1])]

    return df


def extract_tables_from_image(image: Image.Image):
    img_bgr = pil_to_cv(image)
    boxes = detect_table_region(img_bgr)

    dfs = []
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        roi = img_bgr[y:y+h, x:x+w]
        lines = ocr_table_region(roi)
        df = lines_to_dataframe(lines)
        if not df.empty:
            dfs.append((f"Table_{i}", df))

    return dfs


def to_excel_bytes(tables):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary = []

        for sheet_name, df in tables:
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            summary.append({
                "Sheet": safe_name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Column Names": ", ".join(map(str, df.columns))
            })

        if summary:
            pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

    output.seek(0)
    return output.getvalue()


if uploaded_file is not None:
    try:
        all_tables = []

        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            pages = pdf_to_images(pdf_bytes)

            for page_no, page_img in enumerate(pages, start=1):
                page_tables = extract_tables_from_image(page_img)
                for idx, (_, df) in enumerate(page_tables, start=1):
                    all_tables.append((f"Page_{page_no}_T{idx}", df))
        else:
            image = Image.open(uploaded_file)
            img_tables = extract_tables_from_image(image)
            all_tables.extend(img_tables)

        if not all_tables:
            st.warning("No tables detected.")
        else:
            st.success(f"Detected {len(all_tables)} table(s).")

            for name, df in all_tables:
                with st.expander(name, expanded=False):
                    st.dataframe(df, use_container_width=True)

            excel_data = to_excel_bytes(all_tables)
            st.download_button(
                "Download Excel",
                data=excel_data,
                file_name="extracted_tables.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error: {e}")
