import base64
import requests
import json

# 1. Sample PAN Card Text Payload
pan_sample_text = """
INCOME TAX DEPARTMENT
GOVT. OF INDIA
Permanent Account Number Card
ABCDE1234F
Name: RAJESH KUMAR SHARMA
Father's Name: RAMESH CHANDRA SHARMA
Date of Birth: 15/08/1990
"""

# 2. Encode to Base64
b64_payload = base64.b64encode(pan_sample_text.encode("utf-8")).decode("utf-8")

print("==================================================")
print("Sending Base64-Encoded PAN Card Data to FastAPI...")
print("==================================================\n")

# 3. Send POST request to FastAPI
url = "http://127.0.0.1:8000/ner"
try:
    response = requests.post(url, json={"data": b64_payload})
    print("API Response (JSON):")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.ConnectionError:
    print("Could not connect. Make sure 'python main.py' is running!")
