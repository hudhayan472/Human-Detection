# main.py

"""Main application for human detection with UI and DeepStack setup."""

from flask import Flask, request
import base64
import io
import queue
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
import tkinter as tk
from PIL import Image, ImageTk

app = Flask(__name__)

# DeepStack configuration
DEEPSTACK_URL = "http://localhost:5000/v1/vision/detection"  # Change if DeepStack runs elsewhere
CONFIDENCE_THRESHOLD = 0.7

# Queue for passing alarms from Flask thread to UI thread
alarm_queue = queue.Queue()


def wait_for_deepstack():
    """Wait until DeepStack API becomes available on localhost."""

    print("Waiting for DeepStack at http://localhost:5000...")
    for _ in range(30):
        try:
            requests.get(DEEPSTACK_URL, timeout=1)
            print("Connected to DeepStack")
            return True
        except requests.ConnectionError:
            time.sleep(1)
    print("Failed to reach DeepStack")
    return False


class AlarmGUI:
    """Tkinter-based interface for displaying detection alarms."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Human Detection Alarms")

        self.listbox = tk.Listbox(self.root, width=60)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.events = []

    def add_alarm(self, img, details, timestamp):
        self.events.append((img, details, timestamp))
        self.listbox.insert(tk.END, f"{timestamp} - {details}")

    def on_select(self, _event):
        if not self.listbox.curselection():
            return
        idx = self.listbox.curselection()[0]
        img, _details, _timestamp = self.events[idx]
        display = img.copy()
        display.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(display)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img

    def poll_queue(self):
        while True:
            try:
                img, details, timestamp = alarm_queue.get_nowait()
                self.add_alarm(img, details, timestamp)
            except queue.Empty:
                break
        self.root.after(1000, self.poll_queue)

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
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                details = f"Human detected (conf {pred['confidence']:.2f})"
                alarm_queue.put((img.copy(), details, timestamp))
                print(f"ALARM: {details} at {timestamp}")
                return "Human Detected", 200

        print("No human detected")
        return "No Human", 200

    except Exception as e:
        print(f"Error processing alarm: {e}")
        return "Error", 500

if __name__ == '__main__':
    if not wait_for_deepstack():
        raise SystemExit(1)

    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8080, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()

    gui = AlarmGUI()
    gui.poll_queue()
    gui.root.mainloop()
