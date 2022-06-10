import base64
import sys
import time
from io import BytesIO

from flask import Flask, request, jsonify
from consts import ModelSize

app = Flask(__name__)
print("--> Starting DALL-E Server. This might take up to two minutes.")

from dalle_model import DalleModel
dalle_model = None


@app.route("/dalle", methods=["POST"])
def generate_images_api():
    json_data = request.get_json(force=True)
    
    try:
        text_prompt = str(json_data["text"])
        num_images = int(json_data["num_images"])
    except KeyError:
        return jsonify({'error': {'code': 400}, 'message': 'Missing "text" and/or "num_images" keys in JSON data'})
    
    start = time.perf_counter()
    generated_imgs = dalle_model.generate_images(text_prompt, num_images)
    end = time.perf_counter()
    
    unknown_data = json_data.copy()
    
    for key in ('text', 'num_images'):
        unknown_data.pop(key)
    
    js = {'time': {'start': start, 'end': end, 'elapsed': end-start, 'average': (end-start)/num_images}, 'passed': {'text': text_prompt, 'num_images': num_images, 'unknown': unknown_data}}

    generated_images = []
    for img in generated_imgs:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        generated_images.append(img_str)
        
    js['images'] = generated_images

    print(f"Created {num_images} images from text prompt [{text_prompt}]")
    return jsonify(generated_images)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify(success=True)


with app.app_context():
    try:
        dalle_version = ModelSize[sys.argv[2].upper()]
    except (KeyError, IndexError):
        dalle_version = ModelSize.MINI
    dalle_model = DalleModel(dalle_version)
    dalle_model.generate_images("warm-up", 1)
    print("--> DALL-E Server is up and running!")
    print(f"--> Model selected - DALL-E {dalle_version}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(sys.argv[1]), debug=False)
