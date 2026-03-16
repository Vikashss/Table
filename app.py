import io
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from img2table.document import PDF, Image
from img2table.ocr import TesseractOCR


st.set_page_config(page_title="PDF/Image Table to Excel", page_icon="📊", layout="wide")


def clean_text(value):
    if value is None:
        return ""
    value = str(value).replace("\n", " ").replace("\r", " ").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def is_numeric_like(text):
    text = clean_text(text).replace(",", "")
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_good_header(row_values):
    vals = [clean_text(v) for v in row_values]
    non_empty = [v for v in vals if v]

    if len(non_empty) < 2:
        return False

    alpha_count = sum(any(ch.isalpha() for ch in v) for v in non_empty)
    numeric_count = sum(is_numeric_like(v) for v in non_empty)

    return alpha_count >= max(1, numeric_count)


def make_unique_headers(headers):
    final_headers = []
    seen = {}

    for i, h in enumerate(headers, start=1):
        h = clean_text(h)
        if not h:
            h = f"Column_{i}"

        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 1

        final_headers.append(h)

    return final_headers


def dataframe_from_extracted_table(extracted_table):
    df = extracted_table.df.copy()

    # Clean all cells
    df = df.map(clean_text) if hasattr(df, "map") else df.applymap(clean_text)

    # Remove empty rows/columns
    df = df.loc[~(df == "").all(axis=1)]
    df = df.loc[:, ~(df == "").all(axis=0)]

    if df.empty:
        return df

    header_row_index = None
    max_check_rows = min(3, len(df))

    for i in range(max_check_rows):
        if is_good_header(df.iloc[i].tolist()):
            header_row_index = i
            break

    if header_row_index is not None:
        headers = make_unique_headers(df.iloc[header_row_index].tolist())
        df = df.iloc[header_row_index + 1:].copy()
        df.columns = headers
    else:
        df.columns = [f"Column_{i+1}" for i in range(df.shape[1])]

    df.reset_index(drop=True, inplace=True)
    return df


def extract_tables(input_path, lang="eng"):
    ext = Path(input_path).suffix.lower()

    ocr = TesseractOCR(n_threads=1, lang=lang)

    if ext == ".pdf":
        doc = PDF(src=input_path)
    elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        doc = Image(src=input_path)
    else:
        raise ValueError("Unsupported file type. Please upload PDF or image.")

    extracted = doc.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        implicit_columns=True,
        borderless_tables=True,
        min_confidence=50
    )

    tables_output = []
    summary_rows = []

    if isinstance(extracted, dict):
        for page_no, tables in extracted.items():
            for idx, table in enumerate(tables, start=1):
                df = dataframe_from_extracted_table(table)
                if df.empty:
                    continue

                sheet_name = f"Page_{page_no}_T{idx}"[:31]
                tables_output.append((sheet_name, df))

                summary_rows.append({
                    "sheet_name": sheet_name,
                    "source_page": page_no,
                    "table_number": idx,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": ", ".join(map(str, df.columns))
                })

    elif isinstance(extracted, list):
        for idx, table in enumerate(extracted, start=1):
            df = dataframe_from_extracted_table(table)
            if df.empty:
                continue

            sheet_name = f"Image_T{idx}"[:31]
            tables_output.append((sheet_name, df))

            summary_rows.append({
                "sheet_name": sheet_name,
                "source_page": "image",
                "table_number": idx,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": ", ".join(map(str, df.columns))
            })

    summary_df = pd.DataFrame(summary_rows)
    return tables_output, summary_df


def to_excel_bytes(tables_output, summary_df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in tables_output:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

    output.seek(0)
    return output.getvalue()


st.title("📊 PDF/Image Table to Excel")
st.write("Upload a PDF or image file. The app will extract tables, detect column names, and create an Excel file.")

uploaded_file = st.file_uploader(
    "Upload PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)

lang = st.selectbox(
    "OCR Language",
    options=["eng", "hin", "eng+hin"],
    index=0,
    help="Use 'eng+hin' if the table contains both English and Hindi."
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    try:
        with st.spinner("Extracting tables..."):
            tables_output, summary_df = extract_tables(temp_path, lang=lang)

        if not tables_output:
            st.warning("No tables found in the uploaded file.")
        else:
            st.success(f"Found {len(tables_output)} table(s).")

            if not summary_df.empty:
                st.subheader("Summary")
                st.dataframe(summary_df, use_container_width=True)

            st.subheader("Extracted Tables")
            for sheet_name, df in tables_output:
                with st.expander(f"{sheet_name} ({df.shape[0]} rows × {df.shape[1]} columns)", expanded=False):
                    st.dataframe(df, use_container_width=True)

            excel_bytes = to_excel_bytes(tables_output, summary_df)
            output_name = f"{Path(uploaded_file.name).stem}_tables.xlsx"

            st.download_button(
                label="⬇️ Download Excel File",
                data=excel_bytes,
                file_name=output_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
