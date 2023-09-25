from flask import Flask
from flask import request
from flask_cors import CORS
from classify_plants import classify_image

app = Flask(__name__)
CORS(app)

@app.route('/plants/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'plant_image' not in request.files:
            return {
                "success": "false",
                "error": "No file part"
            }
    
        f = request.files['plant_image']
        pred = classify_image(f)
        return {
            "success": "true",
            "prediction": pred,
            "error": "None"
        }