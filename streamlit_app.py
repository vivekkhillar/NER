"""
Streamlit UI for Universal Document & ID Intelligence
Supports Aadhaar Card, PAN Card, Passport, Driving License, Certificates, and Generic PDFs/Images.
"""

import streamlit as st
import json
import base64
import io
from PIL import Image
import pymupdf

# Import backend processing engine from main.py
from main import process_extracted_text, run_rapid_ocr, clean_aadhaar_number, clean_and_correct_pan_number

st.set_page_config(
    page_title="Document Intelligence & NER Studio",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .doc-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        color: #fff;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        margin-bottom: 1rem;
    }
    .info-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .field-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .field-value {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🪪 Universal Document & ID Intelligence</div>', unsafe_allow_html=True)
st.caption("Powered by **RapidOCR** (ONNXRuntime) and **spaCy** (`en_core_web_sm`) • Zero Cloud Dependencies")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("✅ **Model**: `en_core_web_sm` (Local)")
    st.success("✅ **OCR**: `RapidOCR` (ONNX Fast CPU)")
    st.info("Supported: **Aadhaar, PAN, Passport, Driving License, Certificates, Generic PDFs**")
    st.markdown("---")
    st.markdown("### 📌 Document Mapping Guide")
    st.markdown("""
    - **Aadhaar**: 12-digit number, Name, Gender, DOB, Address, C/O
    - **Passport**: Passport No, Given Name, Surname, DOB, Expiry, Father/Mother
    - **PAN Card**: 10-char PAN (with OCR correction), Name, Father Name, DOB
    - **Driving License**: DL No, Name, Address, Validity, Issuing Authority
    """)

# Main Content Tabs
tab_upload, tab_base64, tab_samples = st.tabs(["📁 Upload File (PDF / Image)", "🔤 Base64 Payload Input", "⚡ Quick Test Samples"])

extracted_result = None
file_preview_image = None

# --- TAB 1: File Upload ---
with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image document:",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
        key="file_uploader_widget"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name.lower()

        with st.spinner("Processing document with RapidOCR and spaCy..."):
            extracted_text = ""
            is_image = False
            ocr_used = None

            # 1. Process PDF
            if filename.endswith(".pdf") or file_bytes.startswith(b"%PDF"):
                try:
                    pdf_doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                    extracted_text = "".join([page.get_text() for page in pdf_doc])

                    # Render first page as image preview
                    if len(pdf_doc) > 0:
                        page = pdf_doc[0]
                        pix = page.get_pixmap()
                        file_preview_image = Image.open(io.BytesIO(pix.tobytes("png")))

                        # Scanned PDF fallback
                        if not extracted_text.strip():
                            extracted_text = run_rapid_ocr(pix.tobytes("png"))
                            is_image = True
                            ocr_used = "RapidOCR"
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")

            # 2. Process Image
            elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                try:
                    file_preview_image = Image.open(io.BytesIO(file_bytes))
                    is_image = True
                    ocr_used = "RapidOCR"
                    extracted_text = run_rapid_ocr(file_bytes)
                except Exception as e:
                    st.error(f"Error processing image: {e}")

            # 3. Process Text
            else:
                try:
                    extracted_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    extracted_text = file_bytes.decode("latin-1", errors="ignore")

            if extracted_text.strip():
                extracted_result = process_extracted_text(
                    extracted_text,
                    filename=uploaded_file.name,
                    is_image=is_image,
                    ocr_used=ocr_used
                )


# --- TAB 2: Base64 String Input ---
with tab_base64:
    b64_input = st.text_area(
        "Paste your Base64 encoded document or text:",
        placeholder="Paste Base64 string here (e.g. data:image/png;base64,... or raw Base64)...",
        height=160,
        key="b64_text_area"
    )

    if st.button("🚀 Process Base64 Payload", type="primary", key="btn_process_b64"):
        if not b64_input.strip():
            st.warning("Please enter a Base64 string.")
        else:
            with st.spinner("Decoding Base64 and extracting document data..."):
                raw_b64 = b64_input.strip()
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",")[-1]
                missing_padding = len(raw_b64) % 4
                if missing_padding:
                    raw_b64 += "=" * (4 - missing_padding)

                try:
                    decoded_bytes = base64.b64decode(raw_b64)
                    extracted_text = ""
                    is_image = False
                    ocr_used = None

                    # Check PDF
                    if decoded_bytes.startswith(b"%PDF"):
                        pdf_doc = pymupdf.open(stream=decoded_bytes, filetype="pdf")
                        extracted_text = "".join([page.get_text() for page in pdf_doc])
                        if not extracted_text.strip() and len(pdf_doc) > 0:
                            pix = pdf_doc[0].get_pixmap()
                            extracted_text = run_rapid_ocr(pix.tobytes("png"))
                            file_preview_image = Image.open(io.BytesIO(pix.tobytes("png")))
                            is_image = True
                            ocr_used = "RapidOCR"
                    else:
                        # Check Image
                        try:
                            file_preview_image = Image.open(io.BytesIO(decoded_bytes))
                            extracted_text = run_rapid_ocr(decoded_bytes)
                            is_image = True
                            ocr_used = "RapidOCR"
                        except Exception:
                            # Plain text
                            extracted_text = decoded_bytes.decode("utf-8", errors="ignore")

                    if extracted_text.strip():
                        extracted_result = process_extracted_text(
                            extracted_text,
                            filename="base64_payload",
                            is_image=is_image,
                            ocr_used=ocr_used
                        )
                except Exception as e:
                    st.error(f"Base64 Decoding Error: {e}")


# --- TAB 3: Quick Test Samples ---
with tab_samples:
    st.markdown("Select a sample document payload to immediately test parsing:")
    
    sample_options = {
        "Aadhaar Card (Front OCR)": """667658840093
667658840093
Q6IQS/DOB:29/06/2000
Government of India
Vivek Khillar
gQ&/Male
IssueDate:25/07/2013""",
        "Aadhaar Card (Back with Address & C/O)": """Unique Identification Authority of India
Address:
C/O: BIJAY KUMAR KHILLAR
Plot No 142, Kalinga Nagar, Ghatikia,
Bhubaneswar, Khordha, Odisha - 751030
1947 help@uidai.gov.in www.uidai.gov.in
6676 5884 0093""",
        "PAN Card (with OCR correction O -> 0)": """INCOME TAX DEPARTMENT
GOVT. OF INDIA
Permanent Account Number Card
GRDPKO754H
Name: VIVEK KHILLAR
Father's Name: BIJAY KUMAR KHILLAR
Date of Birth: 29/06/2000
Signature""",
        "Passport (with MRZ)": """PASSPORT
REPUBLIC OF INDIA
Type: P  Country Code: IND  Passport No: Z1234567
Given Name: VIVEK
Surname: KHILLAR
Nationality: INDIAN  Sex: M
Date of Birth: 29/06/2000
Place of Birth: ODISHA
Father / Legal Guardian: BIJAY KUMAR KHILLAR
Mother: MANJU KHILLAR
Date of Issue: 10/01/2020  Date of Expiry: 09/01/2030
P<INDKHILLAR<<VIVEK<<<<<<<<<<<<<<<<<<<<<<<<<
Z1234567<0IND0006295M3001092<<<<<<<<<<<<<<04""",
        "Driving License": """UNION OF INDIA DRIVING LICENCE
Licence No: OD02-20180012345
Name: VIVEK KHILLAR
S/O: BIJAY KUMAR KHILLAR
DOB: 29/06/2000
Address: PLOT NO 142, KALINGA NAGAR, BHUBANESWAR 751030
Valid Till: 28/06/2040
Authorised By: Licensing Authority RTO Bhubaneswar"""
    }

    selected_sample = st.selectbox("Choose Sample:", list(sample_options.keys()))
    sample_text = sample_options[selected_sample]
    
    st.text_area("Sample OCR Text:", value=sample_text, height=130, disabled=True)
    
    if st.button("⚡ Run Sample Test", type="primary", key="btn_run_sample"):
        extracted_result = process_extracted_text(
            sample_text,
            filename=f"sample_{selected_sample.lower().replace(' ', '_')}.txt",
            is_image=False,
            ocr_used="RapidOCR (Simulated)"
        )


# ====================================================================
# Results & JSON Mapping Viewer
# ====================================================================
if extracted_result is not None:
    st.markdown("---")
    doc_data = extracted_result.get("document_data", {})
    doc_type = doc_data.get("document_type", "GENERIC_DOCUMENT")

    col_view_left, col_view_right = st.columns([1, 1])

    # Left Column: Document Preview & Extracted Fields
    with col_view_left:
        st.subheader("📋 Document Structured Data")
        st.markdown(f'<div class="doc-badge">🏷️ {doc_type}</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            
            # Primary Personal Details
            st.markdown("#### 👤 Personal Information")
            st.markdown(f'<div class="field-label">Full Name</div><div class="field-value">{doc_data.get("name") or "—"}</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="field-label">Gender</div><div class="field-value">{doc_data.get("gender") or "—"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-label">Father / Guardian Name</div><div class="field-value">{doc_data.get("father_name") or "—"}</div>', unsafe_allow_html=True)
                if doc_data.get("spouse_name"):
                    st.markdown(f'<div class="field-label">Spouse Name</div><div class="field-value">{doc_data.get("spouse_name")}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="field-label">Date of Birth (DOB)</div><div class="field-value">{doc_data.get("date_of_birth") or "—"}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-label">Mother Name</div><div class="field-value">{doc_data.get("mother_name") or "—"}</div>', unsafe_allow_html=True)

            # Identification Numbers
            st.markdown("#### 🆔 Identification Numbers")
            ids = doc_data.get("identification_numbers", {})
            if ids:
                for k, v in ids.items():
                    st.markdown(f'<div class="field-label">{k.replace("_", " ").upper()}</div><div class="field-value" style="color: #38bdf8;">{v}</div>', unsafe_allow_html=True)
            else:
                st.write("*No specific ID numbers recognized.*")

            # Dates & Validity
            if doc_data.get("issue_date") or doc_data.get("expiry_date"):
                st.markdown("#### 📅 Validity & Dates")
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown(f'<div class="field-label">Issue Date</div><div class="field-value">{doc_data.get("issue_date") or "—"}</div>', unsafe_allow_html=True)
                with d2:
                    st.markdown(f'<div class="field-label">Expiry Date</div><div class="field-value">{doc_data.get("expiry_date") or "—"}</div>', unsafe_allow_html=True)

            # Place & Location Details
            if doc_data.get("place_of_birth") or doc_data.get("place_of_issue"):
                st.markdown("#### 📍 Place Information")
                p1, p2 = st.columns(2)
                with p1:
                    if doc_data.get("place_of_birth"):
                        st.markdown(f'<div class="field-label">Place of Birth</div><div class="field-value">{doc_data.get("place_of_birth")}</div>', unsafe_allow_html=True)
                with p2:
                    if doc_data.get("place_of_issue"):
                        st.markdown(f'<div class="field-label">Place of Issue</div><div class="field-value">{doc_data.get("place_of_issue")}</div>', unsafe_allow_html=True)

            # Address & Authority
            st.markdown("#### 🏠 Address & Authority")
            addr = doc_data.get("address", {})
            st.markdown(f'<div class="field-label">Full Address</div><div class="field-value">{addr.get("full_address") or "—"}</div>', unsafe_allow_html=True)
            if addr.get("pincode"):
                st.markdown(f'<div class="field-label">Pincode</div><div class="field-value">{addr.get("pincode")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="field-label">Authorised / Issued By</div><div class="field-value">{doc_data.get("authorised_by") or "—"}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Show uploaded image preview if available
        if file_preview_image is not None:
            st.markdown("---")
            st.markdown("#### 🖼️ Document Preview")
            st.image(file_preview_image, caption=extracted_result.get("filename"), use_container_width=True)

    # Right Column: Interactive JSON Tree & NER Details
    with col_view_right:
        st.subheader("🔍 Structured JSON Response")
        st.json(extracted_result)

        # Download JSON Button
        json_str = json.dumps(extracted_result, indent=2)
        st.download_button(
            label="💾 Download JSON Result",
            data=json_str,
            file_name=f"extracted_{doc_type.lower()}.json",
            mime="application/json"
        )

        # Extracted Raw Text Expander
        with st.expander("📄 View Extracted Raw Text"):
            st.text(extracted_result.get("extracted_text", ""))

        # SpaCy Named Entities Table
        with st.expander(f"🏷️ View SpaCy Named Entities ({extracted_result.get('total_entities', 0)})"):
            ents = extracted_result.get("entities", [])
            if ents:
                st.dataframe(
                    [{"Text": e["text"], "Label": e["label"], "Meaning": e["description"], "Offsets": f"{e['start_char']}:{e['end_char']}"} for e in ents],
                    use_container_width=True
                )
            else:
                st.write("No named entities detected by generic spaCy model.")
