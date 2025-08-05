# main.py

from flask import Flask, request
import base64
import requests
import xml.etree.ElementTree as ET

app = Flask(__name__)

# DeepStack configuration
DEEPSTACK_URL = "http://localhost:5000/v1/vision/detection"  # Change if DeepStack runs elsewhere
CONFIDENCE_THRESHOLD = 0.7

@app.route('/alarm', methods=['POST'])
def alarm():
    try:
        xml_data = request.data.decode()
        root = ET.fromstring(xml_data)

        # Extract Base64 image from Hikvision XML
        img_b64 = root.findtext('.//base64Pic')
        if not img_b64:
            return "No image found in alarm payload", 400

        img_bytes = base64.b64decode(img_b64)

        # Send to DeepStack AI
        response = requests.post(
            DEEPSTACK_URL,
            files={"image": ("snapshot.jpg", img_bytes, "image/jpeg")}
        )
        result = response.json()

        # Check DeepStack response for human detection
        for pred in result.get("predictions", []):
            if pred["label"] == "person" and pred["confidence"] >= CONFIDENCE_THRESHOLD:
                print("?? HUMAN DETECTED ??")
                return "Human Detected", 200

        print("No human detected")
        return "No Human", 200

    except Exception as e:
        print(f"Error processing alarm: {e}")
        return "Error", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
