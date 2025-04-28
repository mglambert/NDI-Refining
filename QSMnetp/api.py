import numpy as np
import os
import flask
from flask import Flask, request, jsonify
import scipy.io
import io
from custom_inference import inf

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make model inference using the inf() function
    Accepts a 3D volume and returns the inference result
    
    Request format:
    - JSON with volume data in array format
    - Or multipart/form-data with .mat file
    
    Returns:
    - JSON with inference result
    """
    print('we are here 1')
    # try:
    # Check if the request has data
    if not request.data and not request.files:
        return jsonify({
            'status': 'error',
            'message': 'No input data provided'
        }), 400

    # Handle JSON input
    if request.is_json:
        data = request.get_json()

        if 'volume' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Input JSON must contain a "volume" field'
            }), 400

        volume = np.array(data['volume'])

        # Validate input shape
        if len(volume.shape) != 3:
            return jsonify({
                'status': 'error',
                'message': f'Expected 3D volume, got shape {volume.shape}'
            }), 400

    # Handle file upload
    elif request.files and 'file' in request.files:
        file = request.files['file']

        # Check file format
        if not file.filename.endswith('.mat'):
            return jsonify({
                'status': 'error',
                'message': 'Only .mat files are supported'
            }), 400

        # Load .mat file
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)

        try:
            mat_contents = scipy.io.loadmat(in_memory_file)

            # Get the first data array from the .mat file
            # Assuming the first non-metadata field is our volume
            volume = None
            for key in mat_contents.keys():
                if not key.startswith('__'):  # Skip metadata fields
                    volume = mat_contents[key]
                    break

            if volume is None or len(volume.shape) != 3:
                return jsonify({
                    'status': 'error',
                    'message': 'No valid 3D volume found in .mat file'
                }), 400

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error loading .mat file: {str(e)}'
            }), 400

    else:
        return jsonify({
            'status': 'error',
            'message': 'Unsupported content type. Use application/json or multipart/form-data'
        }), 400

    # Process data with inf() function
    result = inf(volume)

    # Return the result
    return jsonify({
        'status': 'success',
        'result': result.tolist()  # Convert numpy array to list for JSON serialization
    })
    
    # except Exception as e:
    #     print('we are here malazo')
    #     return jsonify({
    #         'status': 'error',
    #         'message': f'Error during inference: {str(e)}'
    #     }), 500

if __name__ == '__main__':
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False)
