import streamlit as st
import pandas as pd
import cv2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import io

st.title("PDF Table → Excel Extractor")

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf","png","jpg","jpeg"])

def extract_table_from_image(image):

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(40,40))
    dilate = cv2.dilate(thresh,kernel,iterations=1)

    contours,_ = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    rows=[]

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        if w>500 and h>50:
            roi = img[y:y+h,x:x+w]

            text = pytesseract.image_to_string(roi)

            row = text.split("\n")
            rows.append(row)

    return rows


if uploaded_file:

    tables=[]

    if uploaded_file.type=="application/pdf":

        pages = convert_from_bytes(uploaded_file.read())

        for page in pages:
            rows = extract_table_from_image(page)
            tables.extend(rows)

    else:

        image = Image.open(uploaded_file)
        rows = extract_table_from_image(image)
        tables.extend(rows)

    cleaned=[r for r in tables if len(r)>3]

    df = pd.DataFrame(cleaned)

    st.write("Extracted Table")
    st.dataframe(df)

    buffer = io.BytesIO()

    df.to_excel(buffer,index=False)

    st.download_button(
        "Download Excel",
        buffer.getvalue(),
        "table_output.xlsx"
    )
