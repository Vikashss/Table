import io
import re

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Hybrid PDF Table to Excel", layout="wide")
st.title("Hybrid PDF/Image Table to Excel")

uploaded_file = st.file_uploader(
    "Upload PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)


def clean_text(text):
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", text)


def pdf_to_images(pdf_bytes, dpi=260):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def pil_to_bgr(image):
    image = image.convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def cluster_positions(values, tolerance=10):
    if not values:
        return []

    values = sorted(values)
    groups = [[values[0]]]

    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])

    return [int(sum(g) / len(g)) for g in groups]


def preprocess_for_lines(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    return bw


def detect_table_regions(img_bgr):
    bw = preprocess_for_lines(img_bgr)
    h, w = bw.shape

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 30), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 30)))

    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    table_mask = cv2.add(horizontal, vertical)

    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    merged = cv2.dilate(table_mask, merge_kernel, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = h * w * 0.03
    for c in contours:
        x, y, bw2, bh2 = cv2.boundingRect(c)
        area = bw2 * bh2
        if area > min_area and bw2 > w * 0.3 and bh2 > h * 0.12:
            boxes.append((x, y, bw2, bh2))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def detect_grid_lines(table_bgr):
    bw = preprocess_for_lines(table_bgr)
    h, w = bw.shape

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 25)))

    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    h_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    y_positions = []
    for c in h_contours:
        x, y, bw2, bh2 = cv2.boundingRect(c)
        if bw2 > w * 0.25:
            y_positions.extend([y, y + bh2])

    x_positions = []
    for c in v_contours:
        x, y, bw2, bh2 = cv2.boundingRect(c)
        if bh2 > h * 0.20:
            x_positions.extend([x, x + bw2])

    x_lines = cluster_positions(x_positions, tolerance=10)
    y_lines = cluster_positions(y_positions, tolerance=10)

    x_lines = sorted(set([x for x in x_lines if 0 <= x <= w]))
    y_lines = sorted(set([y for y in y_lines if 0 <= y <= h]))

    if len(x_lines) >= 2:
        if x_lines[0] > 8:
            x_lines = [0] + x_lines
        if w - x_lines[-1] > 8:
            x_lines = x_lines + [w]

    if len(y_lines) >= 2:
        if y_lines[0] > 8:
            y_lines = [0] + y_lines
        if h - y_lines[-1] > 8:
            y_lines = y_lines + [h]

    return x_lines, y_lines


def draw_grid_preview(table_bgr, x_lines, y_lines):
    preview = table_bgr.copy()

    for x in x_lines:
        cv2.line(preview, (x, 0), (x, preview.shape[0]), (0, 0, 255), 2)

    for y in y_lines:
        cv2.line(preview, (0, y), (preview.shape[1], y), (255, 0, 0), 2)

    return bgr_to_rgb(preview)


def parse_manual_lines(text):
    if not text.strip():
        return []
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            try:
                values.append(int(part))
            except ValueError:
                pass
    return sorted(set(values))


def ocr_cell(cell_bgr):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config="--psm 6")
    return clean_text(text)


def grid_to_dataframe(table_bgr, x_lines, y_lines):
    if len(x_lines) < 2 or len(y_lines) < 2:
        return pd.DataFrame()

    rows = []
    for r in range(len(y_lines) - 1):
        y1, y2 = y_lines[r], y_lines[r + 1]
        if y2 - y1 < 8:
            continue

        row = []
        for c in range(len(x_lines) - 1):
            x1, x2 = x_lines[c], x_lines[c + 1]
            if x2 - x1 < 8:
                row.append("")
                continue

            pad = 3
            crop = table_bgr[
                max(0, y1 + pad):max(0, y2 - pad),
                max(0, x1 + pad):max(0, x2 - pad)
            ]

            if crop.size == 0:
                row.append("")
            else:
                row.append(ocr_cell(crop))

        if any(clean_text(v) for v in row):
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    df = pd.DataFrame(rows)
    df = df.loc[~df.apply(lambda r: all(clean_text(v) == "" for v in r), axis=1)]
    df = df.loc[:, ~df.apply(lambda c: all(clean_text(v) == "" for v in c), axis=0)]
    df = df.reset_index(drop=True)
    df.columns = [f"Column_{i+1}" for i in range(df.shape[1])]
    return df


def tables_to_excel_bytes(table_items):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary = []

        for sheet_name, df in table_items:
            safe_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)
            summary.append({
                "Sheet": safe_sheet,
                "Rows": len(df),
                "Columns": len(df.columns),
            })

        if summary:
            pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

    output.seek(0)
    return output.getvalue()


if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/pdf":
            images = pdf_to_images(uploaded_file.read(), dpi=260)
        else:
            images = [Image.open(uploaded_file)]

        page_labels = [f"Page {i+1}" for i in range(len(images))]
        selected_page_label = st.selectbox("Select page", page_labels)
        page_index = page_labels.index(selected_page_label)

        page_image = images[page_index]
        page_bgr = pil_to_bgr(page_image)

        table_regions = detect_table_regions(page_bgr)
        if not table_regions:
            st.error("No table region detected on this page.")
            st.stop()

        region_labels = [f"Table {i+1}" for i in range(len(table_regions))]
        selected_region_label = st.selectbox("Select detected table", region_labels)
        region_index = region_labels.index(selected_region_label)

        x, y, w, h = table_regions[region_index]
        table_bgr = page_bgr[y:y+h, x:x+w]

        auto_x, auto_y = detect_grid_lines(table_bgr)

        st.subheader("Automatic grid preview")
        st.image(draw_grid_preview(table_bgr, auto_x, auto_y), use_container_width=True)

        st.write("You can keep the auto lines, or type your own line positions in pixels.")

        col1, col2 = st.columns(2)
        with col1:
            manual_x_text = st.text_area(
                "Vertical lines (x positions, comma separated)",
                value=",".join(map(str, auto_x)),
                height=120,
            )
        with col2:
            manual_y_text = st.text_area(
                "Horizontal lines (y positions, comma separated)",
                value=",".join(map(str, auto_y)),
                height=120,
            )

        final_x = parse_manual_lines(manual_x_text)
        final_y = parse_manual_lines(manual_y_text)

        st.subheader("Final grid preview")
        st.image(draw_grid_preview(table_bgr, final_x, final_y), use_container_width=True)

        df = grid_to_dataframe(table_bgr, final_x, final_y)

        if df.empty:
            st.error("No cells extracted from the selected grid.")
        else:
            st.success(f"Extracted {len(df)} rows and {len(df.columns)} columns.")
            st.dataframe(df, use_container_width=True)

            excel_data = tables_to_excel_bytes(
                [(f"{selected_page_label}_{selected_region_label}", df)]
            )

            st.download_button(
                "Download Excel",
                data=excel_data,
                file_name="hybrid_table_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error: {e}")
