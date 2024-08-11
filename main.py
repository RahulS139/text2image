# pip install Flask googletrans torch diffusers


from flask import Flask, request, render_template, send_file
from googletrans import Translator
import torch
from diffusers import DiffusionPipeline
import io
import base64
import os

app = Flask(__name__)


class CFG:
    def __init__(self, seed=42, image_gen_steps=35, image_gen_guidance_scale=9):
        self.device = "cuda"
        self.seed = seed
        self.generator = torch.Generator(self.device).manual_seed(seed)
        self.image_gen_steps = image_gen_steps
        self.image_gen_model_id = "stabilityai/stable-diffusion-3-medium"
        self.image_gen_size = (900, 900)
        self.image_gen_guidance_scale = image_gen_guidance_scale


def initialize_model(cfg, token):
    image_gen_model = DiffusionPipeline.from_pretrained(
        cfg.image_gen_model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=token
    )
    image_gen_model = image_gen_model.to(cfg.device)
    return image_gen_model


def generate_image(prompt, model, cfg):
    image = model(
        prompt, num_inference_steps=cfg.image_gen_steps, generator=cfg.generator,
        guidance_scale=cfg.image_gen_guidance_scale
    ).images[0]

    image = image.resize(cfg.image_gen_size)
    return image


def get_translation(text, dest_lang):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text


hugging_face_auth_token = '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'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        seed = int(request.form.get('seed', 42))
        image_gen_steps = int(request.form.get('image_gen_steps', 35))
        image_gen_guidance_scale = float(request.form.get('image_gen_guidance_scale', 9))
        text_to_translate = request.form.get('text_to_translate')

        cfg = CFG(seed=seed, image_gen_steps=image_gen_steps, image_gen_guidance_scale=image_gen_guidance_scale)
        image_gen_model = initialize_model(cfg, hugging_face_auth_token)

        translation = get_translation(text_to_translate, 'en')
        image = generate_image(translation, image_gen_model, cfg)

        img_path = os.path.join('static', 'generated_image.png')
        image.save(img_path)

        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')

        return render_template('index.html', image_data=img_base64)

    return render_template('index.html')


@app.route('/save_image', methods=['POST'])
def save_image():
    img_path = os.path.join('static', 'generated_image.png')
    if os.path.exists(img_path):
        return send_file(img_path, as_attachment=True)
    return 'No image to save', 400


if __name__ == '__main__':
    app.run(debug=True)
