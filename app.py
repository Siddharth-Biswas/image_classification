import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor once
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

def classify_image_blip(image_url):
    """Downloads image and returns a BLIP-generated caption/classification."""
    try:
        response = requests.get(image_url, stream=True, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error classifying image: {e}")
        return "Image Classification Error"

def clean_or_split(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    text = text.replace(' and ', ',').replace(' or ', ',')
    return [t.strip().lower() for t in text.split(',') if t.strip()]

def parse_include(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    text = text.lower()
    if ' and ' in text:
        and_parts = text.split(' and ')
        return [clean_or_split(part) for part in and_parts]
    else:
        return [clean_or_split(text)]

def preprocess_rules(rules_df):
    parsed_rules = []
    for _, row in rules_df.iterrows():
        include = parse_include(row['Include'])
        exclude = clean_or_split(row['Exclude'])
        label = row['Rule']
        parsed_rules.append((include, exclude, label))
    return parsed_rules

def matches_rule(text, include, exclude):
    for and_block in include:
        if not any(word in text for word in and_block):
            return False
    if any(word in text for word in exclude):
        return False
    return True

def classify_products(product_df, rules_df):
    parsed_rules = preprocess_rules(rules_df)
    titles = product_df['TITLE'].str.lower().fillna('')
    text_based_results = []
    image_based_results = []

    progress_bar = st.progress(0)
    total_products = len(product_df)

    for index, title in enumerate(titles):
        matches = []
        for include, exclude, label in parsed_rules:
            if matches_rule(title, include, exclude):
                matches.append(label)
        text_based_results.append(', '.join(matches))

        # Image classification using BLIP
        image_url = product_df['IMAGE_URL'].iloc[index]
        if pd.isna(image_url):
            image_classification = "No Image URL"
        else:
            image_classification = classify_image_blip(image_url)
        image_based_results.append(image_classification)

        progress = (index + 1) / total_products
        progress_bar.progress(progress)

    product_df['mapped_classifications_text'] = text_based_results
    product_df['image_classification'] = image_based_results
    return product_df

# Streamlit UI
st.set_page_config(page_title="Product Classifier", layout="wide")
st.title("🧠 Product Classifier Tool for Flywheel (with BLIP Image Analysis)")
st.markdown("Upload your product details and rules to auto-classify, now using BLIP for smart image understanding!")

product_file = st.file_uploader("📦 Upload Product Details (Excel/CSV)", type=["xlsx", "csv"])
rules_file = st.file_uploader("📋 Upload Classification Rules (Excel/CSV)", type=["xlsx", "csv"])

if product_file and rules_file:
    try:
        product_df = (
            pd.read_excel(product_file)
            if product_file.name.endswith('xlsx')
            else pd.read_csv(product_file)
        )
        rules_df = (
            pd.read_excel(rules_file)
            if rules_file.name.endswith('xlsx')
            else pd.read_csv(rules_file)
        )

        if "TITLE" not in product_df.columns or "IMAGE_URL" not in product_df.columns:
            st.error("Product file must contain 'TITLE' and 'IMAGE_URL' columns.")
            st.stop()
        if not all(col in rules_df.columns for col in ['Rule', 'Include', 'Exclude']):
            st.error("Rules file must contain 'Rule', 'Include', and 'Exclude' columns.")
            st.stop()

    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()

    st.success("✅ Files uploaded successfully!")

    output_df = classify_products(product_df.copy(), rules_df.copy())

    st.subheader("🔍 Preview of Classified Products")
    st.dataframe(output_df, use_container_width=True)

    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Classified CSV", csv, "classified_products.csv", "text/csv")

elif product_file or rules_file:
    st.warning("⚠️ Please upload both the Product and Rules files to proceed.")
