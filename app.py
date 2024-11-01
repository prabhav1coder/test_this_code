from flask import Flask, request, send_file, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will allow all origins by default

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Load GAN model (update path accordingly)
generator_model = tf.keras.models.load_model('generator_model.h5')

# Image Preprocessing and Postprocessing Functions
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize if required by your model
    return np.expand_dims(image_array, axis=0)

def postprocess_image(image_array):
    image_array = (image_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)

@app.route('/process-image', methods=['POST'])
def process_image():
    logger.info('Received image processing request')
    
    if 'image' not in request.files:
        logger.error('No image uploaded!')
        return jsonify({'error': 'No image uploaded!'}), 400

    file = request.files['image']
    image = Image.open(file.stream)

    # Preprocess and generate image with GAN model
    input_tensor = preprocess_image(image)
    logger.info('Preprocessed image')
    output_tensor = generator_model.predict(input_tensor)
    logger.info('Generated image with GAN model')
    output_image = postprocess_image(output_tensor[0])

    # Save image in memory
    image_io = io.BytesIO()
    output_image.save(image_io, format='JPEG')
    image_io.seek(0)

    logger.info('Processed image sent back to client')
    return send_file(image_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5001)
