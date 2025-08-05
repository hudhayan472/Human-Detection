# main.py

from flask import Flask, request
import base64
import requests
import xml.etree.ElementTree as ET
import subprocess
import time
import io
from PIL import Image

app = Flask(__name__)

# DeepStack configuration
DEEPSTACK_URL = "http://localhost:5000/v1/vision/detection"  # Change if DeepStack runs elsewhere
CONFIDENCE_THRESHOLD = 0.7


def start_deepstack():
    """Start DeepStack if it's not already running."""
    try:
        # Check if DeepStack is already reachable
        requests.get(DEEPSTACK_URL, timeout=1)
        print("DeepStack already running")
        return
    except requests.ConnectionError:
        pass

    print("Starting DeepStack...")
    subprocess.Popen(
        ["deepstack", "--VISION-DETECTION", "True"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for DeepStack to become available
    for _ in range(30):
        try:
            requests.get(DEEPSTACK_URL, timeout=1)
            print("DeepStack started")
            return
        except requests.ConnectionError:
            time.sleep(1)
    print("Failed to reach DeepStack")

@app.route('/alarm', methods=['POST'])
def alarm():
    try:
        # Some NVRs send both a snapshot and a clip. Only process the snapshot.
        if 'image' in request.files:
            img_bytes = request.files['image'].read()
        else:
            xml_data = request.data.decode()
            root = ET.fromstring(xml_data)

            # Extract Base64 snapshot from XML and ignore any clips
            img_b64 = root.findtext('.//base64Pic')
            if not img_b64:
                return "No image found in alarm payload", 400

            img_bytes = base64.b64decode(img_b64)

        img = Image.open(io.BytesIO(img_bytes))
        img.save("last_snapshot.jpg")

        # Send to DeepStack AI
        response = requests.post(
            DEEPSTACK_URL,
            files={"image": ("snapshot.jpg", img_bytes, "image/jpeg")}
        )
        result = response.json()

        # Check DeepStack response for human detection
        for pred in result.get("predictions", []):
            if pred["label"] == "person" and pred["confidence"] >= CONFIDENCE_THRESHOLD:
                img.show()
                print("ALARM: HUMAN DETECTED")
                return "Human Detected", 200

        print("No human detected")
        return "No Human", 200

    except Exception as e:
        print(f"Error processing alarm: {e}")
        return "Error", 500

if __name__ == '__main__':
    start_deepstack()
    app.run(host="0.0.0.0", port=8080)
