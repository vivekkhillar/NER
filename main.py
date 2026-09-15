from pathlib import Path
import base64
import io
import re
import spacy
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
from PIL import Image
import pymupdf  # PyMuPDF for PDF text extraction
from rapidocr_onnxruntime import RapidOCR

# 1. Load spaCy model from local directory
MODEL_DIR = Path(__file__).parent / "models" / "en_core_web_sm"
if MODEL_DIR.exists():
    nlp = spacy.load(MODEL_DIR)
else:
    nlp = spacy.load("en_core_web_sm")

# 2. Initialize RapidOCR Engine (ONNX-based, ultra-fast CPU inference)
ocr_engine = RapidOCR()

def run_rapid_ocr(image_input) -> str:
    """Runs RapidOCR on image bytes or array and returns joined text."""
    try:
        result, _ = ocr_engine(image_input)
        if result:
            return "\n".join([line[1] for line in result])
    except Exception as e:
        print(f"RapidOCR Error: {e}")
    return ""


app = FastAPI(
    title="Universal Document & ID Intelligence API (RapidOCR + spaCy)",
    description="Accurate extraction for Aadhaar Card, PAN Card, Passport, Driving License, and any PDF/Image document."
)


# ====================================================================
# 3. ID Number Cleaners & OCR Correction
# ====================================================================
def clean_aadhaar_number(text: str) -> str:
    """Extracts and formats 12-digit Aadhaar number (e.g., 6676 5884 0093)."""
    # Spaced format: 6676 5884 0093
    m1 = re.search(r"\b([2-9]\d{3}\s\d{4}\s\d{4})\b", text)
    if m1:
        return m1.group(1)

    # Continuous 12-digit format: 667658840093
    m2 = re.search(r"\b([2-9]\d{11})\b", text)
    if m2:
        raw = m2.group(1)
        return f"{raw[:4]} {raw[4:8]} {raw[8:]}"

    # Masked format: XXXX XXXX 0093 or **** **** 0093
    m3 = re.search(r"\b([X\*]{4}\s[X\*]{4}\s\d{4})\b", text, re.IGNORECASE)
    if m3:
        return m3.group(1)

    return None


def clean_and_correct_pan_number(text: str) -> str:
    """Extracts 10-character PAN string and corrects common OCR errors (O -> 0)."""
    exact = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b", text)
    if exact:
        return exact.group(1)

    tokens = re.findall(r"\b[A-Za-z0-9]{10}\b", text)
    digit_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
    letter_map = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

    for token in tokens:
        t = token.upper()
        p1, p2, p3 = t[:5], t[5:9], t[9]
        cp1 = "".join(letter_map.get(c, c) for c in p1)
        cp2 = "".join(digit_map.get(c, c) for c in p2)
        cp3 = letter_map.get(p3, p3)
        cand = f"{cp1}{cp2}{cp3}"
        if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", cand):
            return cand
    return None


def extract_gender(text: str) -> str:
    """Extracts Gender (Male / Female / Transgender)."""
    if re.search(r"\b(?:Male|MALE|पुरुष)\b", text, re.IGNORECASE):
        return "Male"
    elif re.search(r"\b(?:Female|FEMALE|महिला)\b", text, re.IGNORECASE):
        return "Female"
    elif re.search(r"\b(?:Transgender|TRANSGENDER)\b", text, re.IGNORECASE):
        return "Transgender"
    return None


# ====================================================================
# 4. Specialized Document Parsers
# ====================================================================

def parse_aadhaar_card(text: str, doc) -> dict:
    """Specialized parser for Aadhaar Card (Front, Back & e-Aadhaar)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    
    aadhaar_no = clean_aadhaar_number(text)
    gender = extract_gender(text)

    # DOB / Year of Birth (multiline supported)
    dob_m = re.search(r"(?:DOB|Date\s*of\s*Birth|जन्म(?:\s*तिथि)?)[^0-9]*?(\d{2}[/-]\d{2}[/-]\d{4})", text, re.IGNORECASE)
    if not dob_m:
        dob_m = re.search(r"(?:Year\s*of\s*Birth|YOB|जन्म\s*वर्ष)[^0-9]*?(\d{4})", text, re.IGNORECASE)
    if not dob_m:
        dob_m = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", text)
    dob = dob_m.group(1) if dob_m else None

    # Issue Date
    issue_m = re.search(r"(?:Issue\s*Date|Issued|IssueDate|VID)[^0-9]*?(\d{2}[/-]\d{2}[/-]\d{4})", text, re.IGNORECASE)
    issue_date = issue_m.group(1) if issue_m else None

    # Name Extraction
    name = None
    for i, line in enumerate(lines):
        # Line following 'Government of India' / 'भारत सरकार'
        if re.search(r"(?:Government\s*of\s*India|भारत\s*सरकार)", line, re.IGNORECASE):
            if i + 1 < len(lines):
                cand = re.sub(r"[^A-Za-z\s\.]", "", lines[i+1]).strip()
                if len(cand) > 2 and not re.search(r"(dob|male|female|birth|issue|india|govt)", cand, re.IGNORECASE):
                    name = cand
                    break
        # Line before 'Male / Female' or 'DOB'
        elif re.search(r"(?:male|female|पुरुष|महिला|dob|birth)", line, re.IGNORECASE):
            if i - 1 >= 0:
                cand = re.sub(r"[^A-Za-z\s\.]", "", lines[i-1]).strip()
                if len(cand) > 2 and not re.search(r"(dob|government|india|birth|issue|govt|authority)", cand, re.IGNORECASE):
                    name = cand
                    break

    # Father / Guardian Name from C/O or S/O
    father_name = None
    co_m = re.search(r"(?:C/O|S/O|W/O|D/O|Care of|Son of|Wife of|Daughter of|आत्मज|पुत्र|पत्नी)[:\s/]+([A-Za-z\s\.]+?)(?:,|\n|$)", text, re.IGNORECASE)
    if co_m:
        father_name = co_m.group(1).strip()

    # Address & Pincode
    addr_m = re.search(r"(?:Address|पता)[:\s/]+([\s\S]*?)(?=\n\s*\n|\b\d{4}\s\d{4}\s\d{4}\b|\b1947\b|\bwww\b|\bhelp\b|$)", text, re.IGNORECASE)
    address = re.sub(r"\s+", " ", addr_m.group(1)).strip() if addr_m else None
    
    pin_m = re.search(r"\b[1-9][0-9]{5}\b", text)
    pincode = pin_m.group(0) if pin_m else None

    return {
        "document_type": "AADHAAR_CARD",
        "name": name,
        "father_name": father_name,
        "mother_name": None,
        "gender": gender,
        "date_of_birth": dob,
        "issue_date": issue_date,
        "identification_numbers": {
            "aadhaar_number": aadhaar_no
        } if aadhaar_no else {},
        "address": {
            "full_address": address,
            "pincode": pincode
        },
        "authorised_by": "Unique Identification Authority of India (UIDAI)"
    }


def parse_mrz_date(yy_mm_dd_str: str, is_expiry: bool = False) -> str:
    """Decodes 6-digit YYMMDD date from Passport MRZ into DD/MM/YYYY."""
    if not yy_mm_dd_str or len(yy_mm_dd_str) != 6 or not yy_mm_dd_str.isdigit():
        return None
    yy = int(yy_mm_dd_str[:2])
    mm = yy_mm_dd_str[2:4]
    dd = yy_mm_dd_str[4:6]
    if is_expiry:
        year = f"20{yy:02d}"
    else:
        # DOB heuristic: if yy > 26 (current year 2026), it's 19yy, else 20yy
        year = f"19{yy:02d}" if yy > 26 else f"20{yy:02d}"
    return f"{dd}/{mm}/{year}"


def parse_passport(text: str, doc) -> dict:
    """Specialized parser for Indian & International Passports (with MRZ support)."""
    # 1. Try Machine Readable Zone (MRZ) first (100% accurate if present)
    mrz1 = re.search(r"P<([A-Z]{3})([A-Z<]+)", text)
    mrz2 = re.search(r"([A-Z0-9]{8,9})<[0-9]([A-Z]{3})([0-9]{6})[0-9]([MFX])([0-9]{6})", text)
    
    pass_no = None
    name = None
    gender = None
    mrz_dob = None
    mrz_expiry = None

    if mrz1 and mrz2:
        names_part = mrz1.group(2).split("<<")
        surname = names_part[0].replace("<", " ").strip()
        given_name = names_part[1].replace("<", " ").strip() if len(names_part) > 1 else ""
        name = f"{given_name} {surname}".strip()
        pass_no = mrz2.group(1).replace("<", "")
        gender = "Male" if mrz2.group(4) == "M" else "Female"
        mrz_dob = parse_mrz_date(mrz2.group(3), is_expiry=False)
        mrz_expiry = parse_mrz_date(mrz2.group(5), is_expiry=True)

    # 2. Standard labeled field extraction for Passport Number
    if not pass_no:
        pn_m = re.search(r"(?:Passport\s*No|Passport\s*Number|पासपोर्ट\s*क्र\.?)[:\s/]+([A-PR-WYa-pr-wy][1-9]\d{6,7})", text, re.IGNORECASE)
        if pn_m:
            pass_no = pn_m.group(1)
        else:
            pn_raw = re.search(r"\b([A-PR-WYa-pr-wy][1-9]\d{6,7})\b", text)
            if pn_raw:
                pass_no = pn_raw.group(1)

    # 3. Standard labeled field extraction for Name
    if not name:
        sn_m = re.search(r"(?:Surname|उपनाम)[:\s/]+([A-Za-z\s]+)", text, re.IGNORECASE)
        gn_m = re.search(r"(?:Given\s*Name(?:\(s\))?|दिया\s*गया\s*नाम)[:\s/]+([A-Za-z\s]+)", text, re.IGNORECASE)
        if gn_m and sn_m:
            given_name = re.sub(r"[^A-Za-z\s]", "", gn_m.group(1)).strip().split("\n")[0]
            surname = re.sub(r"[^A-Za-z\s]", "", sn_m.group(1)).strip().split("\n")[0]
            name = f"{given_name} {surname}".strip()
        elif gn_m:
            name = re.sub(r"[^A-Za-z\s]", "", gn_m.group(1)).strip().split("\n")[0]
        else:
            nm = re.search(r"(?:Name|नाम)[:\s/]+([A-Za-z\s]+)", text, re.IGNORECASE)
            if nm:
                name = re.sub(r"[^A-Za-z\s]", "", nm.group(1)).strip().split("\n")[0]

    # Father / Legal Guardian
    father_m = re.search(r"(?:Name\s*of\s*Father(?:\s*/\s*Legal\s*Guardian)?|Father(?:\s*/\s*Legal\s*Guardian)?(?:'?s?\s*Name)?|S/O|Son of|पिता(?:\s*का\s*नाम)?)[:\s/]+([A-Za-z\s\.]+)", text, re.IGNORECASE)
    father_name = re.sub(r"[^A-Za-z\s\.]", "", father_m.group(1)).strip().split("\n")[0] if father_m else None
    if father_name and len(father_name) < 3:
        father_name = None

    # Mother
    mother_m = re.search(r"(?:Name\s*of\s*Mother|Mother(?:'?s?\s*Name)?|M/O|Mother of|माता(?:\s*का\s*नाम)?)[:\s/]+([A-Za-z\s\.]+)", text, re.IGNORECASE)
    mother_name = re.sub(r"[^A-Za-z\s\.]", "", mother_m.group(1)).strip().split("\n")[0] if mother_m else None
    if mother_name and len(mother_name) < 3:
        mother_name = None

    # Spouse
    spouse_m = re.search(r"(?:Name\s*of\s*Spouse|Spouse(?:'?s?\s*Name)?|Husband(?:'?s?\s*Name)?|Wife(?:'?s?\s*Name)?|W/O|H/O|पति\s*/\s*पत्नी)[:\s/]+([A-Za-z\s\.]+)", text, re.IGNORECASE)
    spouse_name = re.sub(r"[^A-Za-z\s\.]", "", spouse_m.group(1)).strip().split("\n")[0] if spouse_m else None
    if spouse_name and len(spouse_name) < 3:
        spouse_name = None

    # 4. Dates (DOB, Issue Date, Expiry Date with bilingual & multiline support)
    dob_m = re.search(r"(?:Date\s*of\s*Birth|DOB|जन्म(?:\s*तिथि)?)[^0-9]*?(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}\s+[A-Za-z]{3}\s+\d{4})", text, re.IGNORECASE)
    dob = dob_m.group(1) if dob_m else mrz_dob

    issue_m = re.search(r"(?:Date\s*of\s*Issue|Issue\s*Date|Issued|जारी(?:\s*करने\s*की\s*तिथि)?)[^0-9]*?(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}\s+[A-Za-z]{3}\s+\d{4})", text, re.IGNORECASE)
    issue_date = issue_m.group(1) if issue_m else None

    expiry_m = re.search(r"(?:Date\s*of\s*Expiry|Expiry\s*Date|Expiry|Expires|समाप्ति(?:\s*की\s*तिथि)?)[^0-9]*?(\d{2}[/-]\d{2}[/-]\d{4}|\d{2}\s+[A-Za-z]{3}\s+\d{4})", text, re.IGNORECASE)
    expiry_date = expiry_m.group(1) if expiry_m else mrz_expiry

    # Fallback for DOB if still missing: scan any standalone date before MRZ
    if not dob:
        all_dates = re.findall(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", text)
        if all_dates:
            dob = all_dates[0]

    # 5. Place of Birth and Place of Issue
    pob_m = re.search(r"(?:Place\s*of\s*Birth|जन्म\s*स्थान)[:\s/]*\n?\s*([A-Za-z0-9,\.\s-]+?)(?=\n\s*(?:Place|Date|जारी|जन्म|P<|Sex|Nationality|\b[A-Z]{2,}\b:|$))", text, re.IGNORECASE)
    place_of_birth = re.sub(r"[^A-Za-z0-9,\.\s-]", "", pob_m.group(1)).strip().split("\n")[0].strip() if pob_m else None
    if place_of_birth and len(place_of_birth) < 2:
        place_of_birth = None

    poi_m = re.search(r"(?:Place\s*of\s*Issue|जारी\s*करने\s*का\s*स्थान)[:\s/]*\n?\s*([A-Za-z0-9,\.\s-]+?)(?=\n\s*(?:Place|Date|जारी|जन्म|P<|Sex|Nationality|\b[A-Z]{2,}\b:|$))", text, re.IGNORECASE)
    place_of_issue = re.sub(r"[^A-Za-z0-9,\.\s-]", "", poi_m.group(1)).strip().split("\n")[0].strip() if poi_m else None
    if place_of_issue and len(place_of_issue) < 2:
        place_of_issue = None

    # 6. Gender
    if not gender:
        if re.search(r"\b(?:Sex|Gender|लिंग)[:\s/]+(?:M|Male|पुरुष)\b", text, re.IGNORECASE):
            gender = "Male"
        elif re.search(r"\b(?:Sex|Gender|लिंग)[:\s/]+(?:F|Female|महिला)\b", text, re.IGNORECASE):
            gender = "Female"

    # 7. Address & Pincode (from Back Page of Indian Passport)
    addr_m = re.search(r"(?:Address|पता)[:\s/]+([\s\S]*?)(?=\n\s*\n|P<|\bPIN\b|\bFile\b|$)", text, re.IGNORECASE)
    address = re.sub(r"\s+", " ", addr_m.group(1)).strip() if addr_m else None

    # If full_address is absent but place_of_birth / place_of_issue exist on Front page, construct a location summary
    if not address and (place_of_birth or place_of_issue):
        places = []
        if place_of_birth:
            places.append(f"Place of Birth: {place_of_birth}")
        if place_of_issue:
            places.append(f"Place of Issue: {place_of_issue}")
        address = ", ".join(places)

    pin_m = re.search(r"\b[1-9][0-9]{5}\b", text)
    pincode = pin_m.group(0) if pin_m else None

    return {
        "document_type": "PASSPORT",
        "name": name,
        "father_name": father_name,
        "mother_name": mother_name,
        "spouse_name": spouse_name,
        "gender": gender,
        "date_of_birth": dob,
        "place_of_birth": place_of_birth,
        "place_of_issue": place_of_issue,
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "identification_numbers": {
            "passport_number": pass_no
        } if pass_no else {},
        "address": {
            "full_address": address,
            "place_of_birth": place_of_birth,
            "place_of_issue": place_of_issue,
            "pincode": pincode
        },
        "authorised_by": "Government of India, Ministry of External Affairs"
    }


def parse_pan_card(text: str, doc) -> dict:
    """Specialized parser for PAN Card."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    pan_number = clean_and_correct_pan_number(text)

    # DOB multiline supported
    dob_m = re.search(r"(?:DOB|Date\s*of\s*(?:Birth|Eirth)|जन्म(?:\s*तिथि)?)[^0-9]*?(\d{2}[/-]\d{2}[/-]\d{4})", text, re.IGNORECASE)
    if not dob_m:
        dob_m = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", text)
    dob = dob_m.group(1) if dob_m else None

    name = None
    father_name = None

    # 1. Inline match first
    nm = re.search(r"(?:(?:Applicant|Holder)?\s*Name|नाम)[:\s]+([A-Za-z \t\.]+)", text, re.IGNORECASE)
    if nm and not re.search(r"(father|mother|permanent|account|card)", nm.group(0), re.IGNORECASE):
        cand = re.sub(r"[^A-Za-z\s\.]", "", nm.group(1)).strip()
        if len(cand) > 2 and cand.isupper():
            name = cand

    fm = re.search(r"(?:Father[^\w\n\r]*Name|Father'?s?\s*Name|S/O|D/O|पिता(?:\s*का\s*नाम)?)[:\s]+([A-Za-z \t\.]+)", text, re.IGNORECASE)
    if fm:
        cand = re.sub(r"[^A-Za-z\s\.]", "", fm.group(1)).strip()
        if len(cand) > 2 and cand.isupper():
            father_name = cand

    # 2. Multiline fallback
    if not name:
        for i, line in enumerate(lines):
            if re.search(r"(name|नाम)", line, re.IGNORECASE) and not re.search(r"(father|permanent|account|card)", line, re.IGNORECASE):
                if i + 1 < len(lines):
                    cand = re.sub(r"[^A-Za-z\s]", "", lines[i+1]).strip()
                    if len(cand) > 2 and not re.search(r"(father|mother|dob|birth)", cand, re.IGNORECASE):
                        name = cand
                        break

    if not father_name:
        for i, line in enumerate(lines):
            if re.search(r"(father|पिता)", line, re.IGNORECASE):
                if i + 1 < len(lines):
                    cand = re.sub(r"[^A-Za-z\s]", "", lines[i+1]).strip()
                    if len(cand) > 2 and not re.search(r"(mother|dob|birth|date)", cand, re.IGNORECASE):
                        father_name = cand
                        break

    return {
        "document_type": "PAN_CARD",
        "name": name,
        "father_name": father_name,
        "mother_name": None,
        "gender": None,
        "date_of_birth": dob,
        "identification_numbers": {
            "pan_number": pan_number
        } if pan_number else {},
        "address": {
            "full_address": None,
            "pincode": None
        },
        "authorised_by": "Income Tax Department, Government of India"
    }


def parse_generic_document(text: str, doc) -> dict:
    """Universal fallback parser for Driving Licenses, Certificates, Official Letters."""
    ids_found = {}
    
    pan = clean_and_correct_pan_number(text)
    if pan:
        ids_found["pan_number"] = pan

    aadhaar = clean_aadhaar_number(text)
    if aadhaar:
        ids_found["aadhaar_number"] = aadhaar

    dl_m = re.search(r"\b([A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{11})\b", text)
    if dl_m:
        ids_found["driving_license"] = dl_m.group(1).replace(" ", "")

    pass_m = re.search(r"\b([A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9])\b", text)
    if pass_m:
        ids_found["passport_number"] = pass_m.group(1)

    voter_m = re.search(r"\b([A-Z]{3}[0-9]{7})\b", text)
    if voter_m:
        ids_found["voter_id"] = voter_m.group(1)

    generic_m = re.search(r"(?:ID\s*No|Roll\s*No|Reg(?:istration)?\s*No|Ref(?:erence)?\s*No|Certificate\s*No|Account\s*No|Emp(?:loyee)?\s*ID)[:\s]*([A-Za-z0-9\-_/]+)", text, re.IGNORECASE)
    if generic_m:
        ids_found["generic_id"] = generic_m.group(1)

    # Name
    name = None
    nm = re.search(r"(?:(?:Applicant|Student|Candidate|Employee|Holder|Member)?\s*Name|नाम)[:\s]+([A-Za-z \t\.]+)", text, re.IGNORECASE)
    if nm:
        name = re.sub(r"[^A-Za-z\s\.]", "", nm.group(1)).strip()

    # Father
    father_name = None
    fm = re.search(r"(?:Father[^\w\n\r]*Name|Father'?s?\s*Name|S/O|D/O|Son of|Daughter of|पिता(?:\s*का\s*नाम)?)[:\s]+([A-Za-z \t\.]+)", text, re.IGNORECASE)
    if fm:
        father_name = re.sub(r"[^A-Za-z\s\.]", "", fm.group(1)).strip()

    # Mother
    mother_name = None
    mm = re.search(r"(?:Mother[^\w\n\r]*Name|Mother'?s?\s*Name|M/O|Mother of|माता(?:\s*का\s*नाम)?)[:\s]+([A-Za-z \t\.]+)", text, re.IGNORECASE)
    if mm:
        mother_name = re.sub(r"[^A-Za-z\s\.]", "", mm.group(1)).strip()

    # DOB (multiline supported)
    dob_m = re.search(r"(?:DOB|Date\s*of\s*Birth|जन्म(?:\s*तिथि)?)[^0-9]*?(\d{2}[/-]\d{2}[/-]\d{4})", text, re.IGNORECASE)
    if not dob_m:
        dob_m = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", text)
    dob = dob_m.group(1) if dob_m else None

    # Address
    addr_m = re.search(r"(?:Permanent\s*Address|Residential\s*Address|Correspondence\s*Address|Address|पता)[:\s]+([\s\S]*?)(?=\n\s*\n|\bPhone\b|\bMobile\b|\bEmail\b|\bTel\b|\bDate\b|\bAuthori[sz]ed\b|$)", text, re.IGNORECASE)
    address = re.sub(r"\s+", " ", addr_m.group(1)).strip() if addr_m else None

    pin_m = re.search(r"\b[1-9][0-9]{5}\b", text)
    pincode = pin_m.group(0) if pin_m else None

    # Authorised By
    authorised_by = None
    auth_m = re.search(r"(?:Authorised\s*By|Authorized\s*By|Issued\s*By|Issuing\s*Authority|Licensing\s*Authority|Signed\s*By|Signatory|Director|Registrar|Dean)[:\s]+([^\n]+)", text, re.IGNORECASE)
    if auth_m:
        authorised_by = auth_m.group(1).strip()

    t_lower = text.lower()
    if "driving licence" in t_lower or "driving license" in t_lower or "driving_license" in ids_found:
        doc_type = "DRIVING_LICENSE"
    elif "election commission" in t_lower or "voter" in t_lower or "voter_id" in ids_found:
        doc_type = "VOTER_ID"
    elif "certificate" in t_lower or "degree" in t_lower:
        doc_type = "CERTIFICATE"
    else:
        doc_type = "GENERIC_DOCUMENT"

    return {
        "document_type": doc_type,
        "name": name,
        "father_name": father_name,
        "mother_name": mother_name,
        "gender": extract_gender(text),
        "date_of_birth": dob,
        "identification_numbers": ids_found,
        "address": {
            "full_address": address,
            "pincode": pincode
        },
        "authorised_by": authorised_by
    }


def extract_document_fields(text: str, doc) -> dict:
    """Router: Automatically routes text to the best specialized document extractor."""
    t_lower = text.lower()
    
    # 1. Check Passport (MRZ or Keywords or Passport No format)
    if "passport" in t_lower or "republic of india" in t_lower or "p<ind" in t_lower or "p<" in t_lower or "given name" in t_lower or re.search(r"\b[A-PR-WYa-pr-wy][1-9]\d{6,7}\b", text):
        return parse_passport(text, doc)

    # 2. Check Aadhaar Card
    if clean_aadhaar_number(text) or "aadhaar" in t_lower or "uidai" in t_lower or "unique identification" in t_lower or ("government of india" in t_lower and ("male" in t_lower or "female" in t_lower)):
        return parse_aadhaar_card(text, doc)

    # 3. Check PAN Card
    if clean_and_correct_pan_number(text) or "permanent account number" in t_lower or "income tax department" in t_lower:
        return parse_pan_card(text, doc)

    # 4. Generic Document / Driving License / Certificate
    return parse_generic_document(text, doc)


def process_extracted_text(text: str, filename: str = None, is_image: bool = False, ocr_used: str = None):
    doc = nlp(text)
    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "description": spacy.explain(ent.label_),
            "start_char": ent.start_char,
            "end_char": ent.end_char
        }
        for ent in doc.ents
    ]

    structured_data = extract_document_fields(text, doc)

    return {
        "filename": filename,
        "is_image_or_scanned": is_image,
        "ocr_engine": ocr_used,
        "document_data": structured_data,
        "extracted_text": text.strip(),
        "total_entities": len(entities),
        "entities": entities
    }


# ====================================================================
# 5. API Endpoints
# ====================================================================
class NERRequest(BaseModel):
    data: str  # Base64 string of text OR image OR PDF


# --- Endpoint 1: Base64 Payload ---
@app.post("/ner", summary="Process Base64 Payload (Text / Image / PDF)")
def extract_from_base64(payload: NERRequest):
    try:
        raw_b64 = payload.data.strip()
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[-1]
            
        missing_padding = len(raw_b64) % 4
        if missing_padding:
            raw_b64 += "=" * (4 - missing_padding)

        decoded_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64: {str(e)}")

    # Check if PDF
    if decoded_bytes.startswith(b"%PDF"):
        try:
            pdf_doc = pymupdf.open(stream=decoded_bytes, filetype="pdf")
            extracted_text = "".join([page.get_text() for page in pdf_doc])
            
            # Scanned PDF RapidOCR fallback
            if not extracted_text.strip() and len(pdf_doc) > 0:
                page = pdf_doc[0]
                pix = page.get_pixmap()
                extracted_text = run_rapid_ocr(pix.tobytes("png"))
                return process_extracted_text(extracted_text, filename="document.pdf", is_image=True, ocr_used="RapidOCR")
                
            return process_extracted_text(extracted_text, filename="document.pdf", is_image=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    # Check if Image using RapidOCR
    try:
        Image.open(io.BytesIO(decoded_bytes))
        extracted_text = run_rapid_ocr(decoded_bytes)
        if extracted_text:
            return process_extracted_text(extracted_text, filename="image.png", is_image=True, ocr_used="RapidOCR")
    except Exception:
        pass

    # Plain Text
    try:
        extracted_text = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        extracted_text = decoded_bytes.decode("latin-1", errors="ignore")

    return process_extracted_text(extracted_text, filename="text_input", is_image=False)


# --- Endpoint 2: Direct File Upload (Swagger UI /docs) ---
@app.post("/upload-file", summary="Upload any PDF or Image directly (Swagger UI /docs)")
async def upload_document_file(file: UploadFile = File(...)):
    """
    Upload any `.pdf`, `.png`, `.jpg`, `.jpeg`, or `.txt` file directly from Swagger `/docs`.
    Uses RapidOCR for ultra-fast ONNX image/scanned PDF text extraction.
    """
    file_bytes = await file.read()
    filename = file.filename.lower()

    extracted_text = ""
    is_image = False
    ocr_used = None

    # 1. PDF File
    if filename.endswith(".pdf") or file_bytes.startswith(b"%PDF"):
        try:
            pdf_doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            extracted_text = "".join([page.get_text() for page in pdf_doc])
            
            # Scanned PDF RapidOCR fallback
            if not extracted_text.strip() and len(pdf_doc) > 0:
                page = pdf_doc[0]
                pix = page.get_pixmap()
                extracted_text = run_rapid_ocr(pix.tobytes("png"))
                is_image = True
                ocr_used = "RapidOCR"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

    # 2. Image File (PNG, JPG, JPEG, WEBP)
    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        try:
            is_image = True
            ocr_used = "RapidOCR"
            extracted_text = run_rapid_ocr(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    # 3. Plain Text File
    else:
        try:
            extracted_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = file_bytes.decode("latin-1", errors="ignore")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

    return process_extracted_text(extracted_text, filename=file.filename, is_image=is_image, ocr_used=ocr_used)


# --- Endpoint 3: Health Check ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_DIR),
        "ocr_engine": "RapidOCR (ONNXRuntime)"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
