from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


MODEL_PATH = 'waste_classification_model.h5'
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    return img_array

def classify_image(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    score = float(prediction[0][0])
    
    
    is_biodegradable = score < 0.5
    result = {
        'classification': 'Biodegradable' if is_biodegradable else 'Non-biodegradable',
        'confidence': round((1 - score) * 100, 2) if is_biodegradable else round(score * 100, 2)
    }
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        
        result = classify_image(file_path)
        
       
        return jsonify({
            'classification': result['classification'],
            'confidence': result['confidence'],
            'image_path': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)