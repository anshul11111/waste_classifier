import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template
model = tf.keras.models.load_model("waste_classification_model.keras")

labels = ['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 'cardboard_packaging','clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 'eggshells', 'food_waste',
'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars', 'magazines', 'newspaper',
'office_paper', 'paper_cups', 'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 'plastic_water_bottles',
'shoes', 'steel_food_cans', 'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags'
]


app = Flask(__name__)
camera = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """ Resize, normalize, and reshape the frame to match model input """
    frame = cv2.resize(frame, (128, 128))  
    frame = frame / 255.0  
    frame = np.expand_dims(frame, axis=0)  
    return frame

def generate_frames():
    """Captures frames, predicts waste category, and overlays label with bounding box"""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = preprocess_frame(rgb_frame)
            predictions = model.predict(processed_frame)
            predicted_class = np.argmax(predictions)
            label = labels[predicted_class]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, f"{label}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """ Video Streaming Home Page """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """ Return the video feed response """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)



