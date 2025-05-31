import streamlit as st
import base64
import json
import pandas as pd
from openai import OpenAI
import re
from PyPDF2 import PdfReader

import logging

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "your-api-key")
MODEL = "gpt-4.1-mini" # Use a smaller model for cost efficiency

CATEGORY_GUIDE = {
    "Transport": ["train ticket", "taxi", "bus fare", "flight"],
    "Parking": ["parking ticket", "parking fee", "garage"],
    "Meals": ["restaurant", "cafe", "food", "drinks"],
    "Accommodation": ["hotel", "bnb"],
    "Other": ["miscellaneous", "other expenses", "unknown"],
}
CATEGORY_LIST = list(CATEGORY_GUIDE.keys())

FIELDS = {
    "filename": "",
    "date": "",
    "vendor": "",
    "amount": 0.0,
    "currency": "",
    "category": "",
    "notes": ""
}

client = OpenAI(api_key=OPENAI_API_KEY)

# --- FUNCTIONS ---

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)

def encode_image_base64(image):
    return base64.b64encode(image.getvalue()).decode("utf-8")

PROMPT_TEMPLATE = """
You are a system that extracts structured data from receipts (provided as text).

Respond in strict JSON format with these fields:
{response_scheme}

Choose the category from the following guide:
{category_guide}

{extra_text}

Only respond with valid JSON.
"""

def build_prompt_with_text(raw_text):
    prompt = PROMPT_TEMPLATE.format(
        response_scheme=json.dumps(FIELDS, indent=2),
        category_guide=json.dumps(CATEGORY_GUIDE, indent=2),
        extra_text="Here is the raw text from a single receipt:\n\"\"\"\n" + raw_text + "\n\"\"\""
    )
    return prompt

def build_prompt():
    prompt = PROMPT_TEMPLATE.format(
        response_scheme=json.dumps(FIELDS, indent=2),
        category_guide=json.dumps(CATEGORY_GUIDE, indent=2),
        extra_text="Please analyze the receipt image and extract structured data."
    )
    return prompt

def call_openai_and_parse(prompt, file_name, image_data=None):
    """
    Calls OpenAI API with the given prompt.
    If image_data is provided, sends as image; otherwise, as text.
    Returns parsed JSON or None on error.
    """
    if image_data:
        # For images
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": "low"
                }}
            ]
        }]
    else:
        # For PDFs (with selectable text)
        messages = [
            {"role": "user", "content": prompt}
        ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000
    )

    raw = response.choices[0].message.content
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        result = json.loads(cleaned)
        result["filename"] = file_name
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error for {file_name}: {e}")
        logger.debug(f"Response was:\n{raw}")
        logger.debug("Please check the file format and content.")
        return None

def extract_data_from_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    if not text.strip():
        logger.debug(f"No selectable text found in {pdf_file.name}. Consider uploading as image instead.")
        return None
    prompt = build_prompt_with_text(text)
    return call_openai_and_parse(prompt, pdf_file.name)

def extract_data_from_image(image_file):
    image_data = encode_image_base64(image_file)
    prompt = build_prompt()
    return call_openai_and_parse(prompt, image_file.name, image_data=image_data)

def extract_data_from_receipt(receipt_file):
    if receipt_file.name.lower().endswith('.pdf'):
        return extract_data_from_pdf(receipt_file)
    elif receipt_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
        return extract_data_from_image(receipt_file)
    else:
        logger.error(f"Unsupported file type: {receipt_file.name}")
        return None

# --- STREAMLIT UI ---

uploaded_files = st.file_uploader("Upload receipt images", accept_multiple_files=True)

if "receipts" not in st.session_state:
    st.session_state.receipts = []
if "previews" not in st.session_state:
    st.session_state.previews = {}

# Extract data
if uploaded_files:
    for file in uploaded_files:
        if any(r["filename"] == file.name for r in st.session_state.receipts):
            continue
        with st.spinner(f"Analyzing {file.name}..."):
            data = extract_data_from_receipt(file)
            if data:
                st.session_state.receipts.append(data)
                st.session_state.previews[file.name] = file
            else:
                st.error(f"Could not extract data from {file.name}")
                # TODO: Add fallback if PDF is a scanned image

# Show data editor
if st.session_state.receipts:
    df = pd.DataFrame(st.session_state.receipts)

    # Display the parsed receipt data
    st.subheader("🧾 Receipts")
    st.dataframe(
        df,
        use_container_width=True,
        key="receipt_table"
    )

    # Image preview
    with st.expander("🔍 Receipt Preview"):
        selected_index = st.number_input(
            "Select row to preview image", min_value=0,
            max_value=len(df) - 1,
            step=1, value=0
        )
        preview_filename = df.iloc[selected_index]["filename"]
        receipt_file = st.session_state.previews.get(preview_filename)
        if receipt_file:
            if receipt_file.name.lower().endswith('.pdf'):
                st.warning("PDF previews are not supported yet. Please download to view.")
                receipt_file.seek(0)
                st.download_button(
                    label="Download PDF",
                    data=receipt_file.read(),
                    file_name=receipt_file.name,
                    mime="application/pdf"
                )            
            else:
                st.image(receipt_file, caption=preview_filename, use_container_width=True)
