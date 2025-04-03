import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template, request, jsonify

model = tf.keras.models.load_model("waste_classification_model.keras")
labels = ['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 
          'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 
          'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers', 
          'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups', 
          'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers', 
          'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 
          'plastic_water_bottles', 'shoes', 'steel_food_cans', 'styrofoam_cups', 
          'styrofoam_food_containers', 'tea_bags']





#flask
app = Flask(__name__)
camera = cv2.VideoCapture(0)


def preprocess_frame(frame):
    """ Resize, normalize, and reshape the frame for model input """
    frame = cv2.resize(frame, (128, 128))  
    frame = frame / 255.0  
    frame = np.expand_dims(frame, axis=0)  
    return frame




def detect_and_label(frame):
    """ Detect object in the frame, draw bounding box, and classify it """
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
        if w > 30 and h > 30:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, label, (x, max(y - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame, label

def generate_frames():
    """ Captures frames, predicts waste category, and overlays label with bounding box """
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, _ = detect_and_label(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """ Render the video streaming page with upload option """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """ Return the video feed response """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_image():
    """ Handle image upload and classify the waste """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image, label = detect_and_label(image)
    _, buffer = cv2.imencode(".jpg", image)
    image_bytes = buffer.tobytes()
    return Response(image_bytes, content_type='image/jpeg', headers={'label': label})

if __name__ == "__main__":
    app.run(debug=True)



# @app.route('/upload', methods=['POST'])
# def upload_image():
#     """ Handle image from ESP32-CAM and classify it """
#     global latest_prediction
#     image_bytes = request.data  # Read raw image data from ESP32
#     image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

#     if image is None:
#         return jsonify({'error': 'Failed to process image'})

#     image, label = detect_and_label(image)
#     latest_prediction = label  # Update latest prediction

#     _, buffer = cv2.imencode(".jpg", image)
#     return Response(buffer.tobytes(), content_type='image/jpeg', headers={'label': label})
