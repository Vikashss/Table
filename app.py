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

st.set_page_config(page_title="Any Table PDF to Excel", layout="wide")
st.title("Any Table PDF/Image to Excel")

uploaded_file = st.file_uploader(
    "Upload PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)


def clean_text(text):
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", text)


def pdf_to_images(pdf_bytes, dpi=240):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def pil_to_bgr(image):
    image = image.convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def get_table_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Better binarization for scanned docs
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    table_mask = cv2.add(horizontal, vertical)
    return table_mask


def find_table_boxes(table_mask, img_shape):
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img_shape[:2]
    min_area = (h * w) * 0.02

    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area > min_area and bw > 100 and bh > 60:
            boxes.append((x, y, bw, bh))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def merge_nearby_positions(values, tolerance=12):
    if not values:
        return []

    values = sorted(values)
    groups = [[values[0]]]

    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])

    return [int(sum(group) / len(group)) for group in groups]


def detect_cells_from_table(table_roi_bgr, table_roi_mask):
    contours, hierarchy = cv2.findContours(table_roi_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h, w = table_roi_mask.shape[:2]
    cell_boxes = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        # likely cell size
        if 200 < area < (h * w * 0.9) and bw > 15 and bh > 15:
            cell_boxes.append((x, y, bw, bh))

    # remove very large parent boxes that are actually full table boxes
    filtered = []
    for box in cell_boxes:
        x, y, bw, bh = box
        if bw < w * 0.98 and bh < h * 0.98:
            filtered.append(box)

    # deduplicate by approximate coordinates
    unique = []
    seen = set()
    for x, y, bw, bh in filtered:
        key = (round(x / 8), round(y / 8), round(bw / 8), round(bh / 8))
        if key not in seen:
            seen.add(key)
            unique.append((x, y, bw, bh))

    return sorted(unique, key=lambda b: (b[1], b[0]))


def build_grid(cell_boxes):
    if not cell_boxes:
        return [], [], {}

    xs = []
    ys = []

    for x, y, w, h in cell_boxes:
        xs.append(x)
        ys.append(y)

    col_positions = merge_nearby_positions(xs, tolerance=15)
    row_positions = merge_nearby_positions(ys, tolerance=15)

    position_map = {}
    for x, y, w, h in cell_boxes:
        col_idx = min(range(len(col_positions)), key=lambda i: abs(col_positions[i] - x))
        row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - y))
        position_map[(row_idx, col_idx)] = (x, y, w, h)

    return row_positions, col_positions, position_map


def ocr_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config="--psm 6")
    return clean_text(text)


def table_to_dataframe(table_roi_bgr, table_roi_mask):
    cell_boxes = detect_cells_from_table(table_roi_bgr, table_roi_mask)

    if not cell_boxes:
        return pd.DataFrame()

    row_positions, col_positions, position_map = build_grid(cell_boxes)

    if not row_positions or not col_positions:
        return pd.DataFrame()

    data = []
    for r in range(len(row_positions)):
        row_data = []
        for c in range(len(col_positions)):
            if (r, c) in position_map:
                x, y, w, h = position_map[(r, c)]

                pad = 2
                cell = table_roi_bgr[
                    max(0, y + pad):max(0, y + h - pad),
                    max(0, x + pad):max(0, x + w - pad)
                ]

                if cell.size == 0:
                    row_data.append("")
                else:
                    row_data.append(ocr_cell(cell))
            else:
                row_data.append("")
        data.append(row_data)

    df = pd.DataFrame(data)

    # remove fully blank rows and columns
    df = df.loc[~(df.apply(lambda row: all(clean_text(v) == "" for v in row), axis=1))]
    if not df.empty:
        df = df.loc[:, ~(df.apply(lambda col: all(clean_text(v) == "" for v in col), axis=0))]

    df = df.reset_index(drop=True)
    df.columns = [f"Column_{i+1}" for i in range(df.shape[1])]
    return df


def extract_tables_from_image(image):
    img_bgr = pil_to_bgr(image)
    table_mask = get_table_mask(img_bgr)
    table_boxes = find_table_boxes(table_mask, img_bgr.shape)

    tables = []
    for idx, (x, y, w, h) in enumerate(table_boxes, start=1):
        roi_bgr = img_bgr[y:y+h, x:x+w]
        roi_mask = table_mask[y:y+h, x:x+w]

        df = table_to_dataframe(roi_bgr, roi_mask)
        if not df.empty:
            tables.append((f"Table_{idx}", df))

    return tables


def save_tables_to_excel(all_tables):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_rows = []

        for sheet_name, df in all_tables:
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

            summary_rows.append({
                "Sheet": safe_name,
                "Rows": len(df),
                "Columns": len(df.columns)
            })

        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

    output.seek(0)
    return output.getvalue()


if uploaded_file is not None:
    try:
        all_tables = []

        if uploaded_file.type == "application/pdf":
            images = pdf_to_images(uploaded_file.read(), dpi=240)

            for page_no, image in enumerate(images, start=1):
                page_tables = extract_tables_from_image(image)
                for table_name, df in page_tables:
                    all_tables.append((f"Page_{page_no}_{table_name}", df))
        else:
            image = Image.open(uploaded_file)
            img_tables = extract_tables_from_image(image)
            all_tables.extend(img_tables)

        if not all_tables:
            st.error("No tables detected.")
        else:
            st.success(f"Detected {len(all_tables)} table(s).")

            for name, df in all_tables:
                with st.expander(name, expanded=False):
                    st.dataframe(df, use_container_width=True)

            excel_bytes = save_tables_to_excel(all_tables)

            st.download_button(
                "Download Excel",
                data=excel_bytes,
                file_name="all_detected_tables.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error: {e}")
