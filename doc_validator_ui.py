#!/usr/bin/env python3
"""
Streamlit Demo UI for Azure Document Validator (Polished Version)
- Clean interface for management demo
- No validator script path field
- No applicant name line
- Simple overall summary text (no big PASS/FAIL headline)
"""

import streamlit as st
import subprocess
import json
from pathlib import Path
import fitz
from PIL import Image
from io import BytesIO
import pandas as pd

# ------------------------
# Helper functions
# ------------------------

def run_validator(input_dir, validator_script="azure_doc_validator.py"):
    """Run azure_doc_validator.py exactly as from terminal."""
    output_path = Path(input_dir) / "document_validation_results.json"
    cmd = [
        "python",
        validator_script,
        "--input_dir", str(input_dir),
        "--output", str(output_path)
    ]

    st.write("🟢 Running:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        st.text_area("📜 Validator Output Log", proc.stdout + "\n" + proc.stderr, height=200)
        if proc.returncode == 0 and output_path.exists():
            st.success(f"✅ Validation complete. Results saved to {output_path}")
            return output_path
        else:
            st.error("❌ Validator failed or no JSON generated.")
            return None
    except Exception as e:
        st.error(f"Error running validator: {e}")
        return None


def load_json_results(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not read JSON: {e}")
        return None


def render_pdf_pages(pdf_path):
    """Render all pages of PDF as PIL images."""
    doc = fitz.open(pdf_path)
    imgs = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(BytesIO(pix.tobytes("png")))
        imgs.append(img)
    return imgs


def style_status(val):
    if val == "PASS":
        return "✅ PASS"
    elif val == "FAIL":
        return "❌ FAIL"
    elif val == "REVIEW":
        return "⚠️ REVIEW"
    else:
        return val


# ------------------------
# Streamlit Layout
# ------------------------

st.set_page_config(page_title="Azure Document Validator", layout="wide")
st.title("🔎 Azure Document Validator — Demo Dashboard")

st.write("""
Upload the folder path containing PDFs and click **Run Validator**  
to execute the existing Azure Document Validator script and see results visually.
""")

# Initialize session state
if "validation_done" not in st.session_state:
    st.session_state.validation_done = False
if "result_json" not in st.session_state:
    st.session_state.result_json = None
if "input_dir" not in st.session_state:
    st.session_state.input_dir = None
if "data" not in st.session_state:
    st.session_state.data = None

# --- Inputs (only directory now) ---
input_dir = st.text_input("📂 Enter directory path containing PDFs:", value="input_docs")
run_button = st.button("▶️ Run Validator")

# --- Run validator only on button click ---
if run_button:
    if not Path(input_dir).exists():
        st.error(f"Input directory not found: {input_dir}")
    else:
        with st.spinner("Running validator... please wait (Azure OCR)..."):
            result_json = run_validator(input_dir, validator_script="azure_doc_validator.py")
        if result_json and Path(result_json).exists():
            data = load_json_results(result_json)
            if data:
                st.session_state.validation_done = True
                st.session_state.result_json = result_json
                st.session_state.input_dir = input_dir
                st.session_state.data = data
        else:
            st.error("Validator failed or no JSON generated.")

# --- Display results if available ---
if st.session_state.validation_done and st.session_state.data:
    data = st.session_state.data
    input_dir = st.session_state.input_dir
    results = data.get("results", [])

    df = pd.DataFrame(results)
    if not df.empty:
        df["File"] = df["file"].apply(lambda x: Path(x).name)
        df["Score"] = df["score"].apply(lambda x: f"{round(x)} %")
        df["Status"] = df["status"].apply(style_status)
        df.rename(columns={"detected_name": "Detected Name"}, inplace=True)
        df = df[["File", "Detected Name", "Score", "Status"]]

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Simple summary (no big PASS/FAIL headline)
        pass_count = (df["Status"].str.contains("PASS")).sum()
        fail_count = (df["Status"].str.contains("FAIL")).sum()
        total = len(df)
        st.write(f"**Summary:** {pass_count} PASS  |  {fail_count} FAIL  (out of {total})")

        st.markdown("---")
        st.subheader("📘 Document Preview")

        # Dropdown for PDF preview
        selected_file = st.selectbox(
            "Select a document to preview:",
            options=df["File"].tolist(),
            key="pdf_selector"
        )

        if selected_file:
            pdf_path = Path(input_dir) / selected_file
            if pdf_path.exists():
                pages = render_pdf_pages(pdf_path)
                for i, p in enumerate(pages):
                    st.image(p, caption=f"Page {i+1}", use_column_width=True)
            else:
                st.warning(f"{selected_file} not found in directory.")
else:
    st.info("Enter your folder path and click 'Run Validator' to generate results.")
