# 🪪 Universal Document & ID Intelligence Studio (RapidOCR + spaCy)

An offline, high-speed document intelligence application and API capable of processing Base64 payloads, PDFs, and images (`.png`, `.jpg`, `.jpeg`, `.webp`). Extracts structured JSON and named entities using **RapidOCR (ONNX)** and **spaCy** (`en_core_web_sm`).

---

## 🌟 Key Features
- 🪪 **Smart Document Routing**: Automatically detects **Aadhaar Card**, **PAN Card**, **Passport**, **Driving License**, and **Generic Documents**.
- 🔍 **High-Speed ONNX OCR**: RapidOCR runs locally on CPU without PyTorch/CUDA bloat.
- ⚡ **Structured JSON Extraction**:
  - **Full Name**, **Father / Guardian Name**, **Mother Name**, **Spouse Name**, **Gender**, **DOB**
  - **ID Numbers**: Aadhaar (12-digit formatted), PAN (with OCR error correction e.g. `O` -> `0`), Passport No, Driving License, Voter ID
  - **Dates**: Issue Date, Expiry Date
  - **Address**: Full Address & Pincode
  - **Authorised / Issuing Authority**
- 💻 **Interactive Streamlit UI**: File uploader, live document preview, structured info cards, interactive JSON explorer, and one-click JSON download.
- 🚀 **FastAPI Backend**: `/ner` (Base64 string) and `/upload-file` (direct Swagger `/docs` upload).
- 🔒 **100% Offline**: Pre-packaged raw model directory in `./models/en_core_web_sm` (no internet access needed).

---

## 📁 Project Structure

```
NER/
├── main.py                 # FastAPI backend & Document Extraction Engine
├── streamlit_app.py        # Streamlit Web UI & Interactive JSON Viewer
├── test_client.py          # Python test script for API requests
├── requirements.txt        # Python dependencies
├── models/
│   └── en_core_web_sm/     # Local raw spaCy model directory
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit UI
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run FastAPI Server (Optional for API integrations)
```bash
python main.py
```
- API Endpoint: `http://127.0.0.1:8000/ner`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

---

## 📡 API Usage (cURL)

### Base64 POST Request:
```bash
curl -X POST "http://127.0.0.1:8000/ner" \
     -H "Content-Type: application/json" \
     -d "{\"data\": \"<BASE64_STRING_OR_DATA_URL>\"}"
```

### Direct File Upload POST Request:
```bash
curl -X POST "http://127.0.0.1:8000/upload-file" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@Adhar_front.jpeg"
```

---

## 📋 Sample Structured JSON Output

```json
{
  "filename": "Adhar_front.jpeg",
  "is_image_or_scanned": true,
  "ocr_engine": "RapidOCR",
  "document_data": {
    "document_type": "AADHAAR_CARD",
    "name": "Vivek Khillar",
    "father_name": null,
    "mother_name": null,
    "gender": "Male",
    "date_of_birth": "29/06/2000",
    "issue_date": "25/07/2013",
    "identification_numbers": {
      "aadhaar_number": "6676 5884 0093"
    },
    "address": {
      "full_address": null,
      "pincode": null
    },
    "authorised_by": "Unique Identification Authority of India (UIDAI)"
  },
  "extracted_text": "...",
  "total_entities": 1,
  "entities": [ ... ]
}
```
