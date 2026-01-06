from flask import Flask, request, render_template, send_file, jsonify
import os, json, subprocess
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)

# === CONFIG ===
BASE_DIR = os.path.abspath('modnet_workspace')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
BG_DIR = os.path.join(BASE_DIR, 'background')
CKPT_PATH = os.path.abspath('MODNet/pretrained/modnet_photographic_portrait_matting.ckpt')
POSITION_FILE = 'foreground_position.json'

# === SETUP ===
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BG_DIR, exist_ok=True)

def extract_subject(image_path, matte_path, save_path):
    img = np.array(Image.open(image_path).convert('RGB'))
    matte = np.array(Image.open(matte_path).convert('L')) / 255.0
    matte = np.stack([matte] * 3, axis=-1)
    fg = (img * matte).astype(np.uint8)
    alpha = (matte[..., 0] * 255).astype(np.uint8)
    fg_rgba = np.dstack((fg, alpha))
    Image.fromarray(fg_rgba).save(save_path)

def remove_green_spill(image_path, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    bgr = img[:, :, :3]
    alpha = img[:, :, 3] if img.shape[2] == 4 else None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    hsv[:, :, 1] = np.where(mask > 0, hsv[:, :, 1] * 0.5, hsv[:, :, 1])
    cleaned = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if alpha is not None:
        cleaned = cv2.merge((cleaned, alpha))
    cv2.imwrite(save_path, cleaned)

def save_position(bg_name, position):
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[bg_name] = position
    with open(POSITION_FILE, 'w') as f:
        json.dump(data, f)

def load_position(bg_name):
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, 'r') as f:
            data = json.load(f)
        return data.get(bg_name)
    return None

def composite_with_position(fg_path, bg_path, x, y, scale, save_path):
    fg = Image.open(fg_path).convert("RGBA")
    bg = Image.open(bg_path).convert("RGBA")
    fg = fg.resize((int(fg.width * scale), int(fg.height * scale)))
    result = bg.copy()
    result.paste(fg, (int(x), int(y)), fg)
    result.save(save_path)

@app.route('/')
def index():
    return render_template('admin.html')

@app.route('/upload', methods=['POST'])
def upload():
    subject = request.files['subject']
    background = request.files['background']
    bg_name = request.form['bg_name']

    subject_path = os.path.join(INPUT_DIR, 'subject.jpg')
    background_path = os.path.join(BG_DIR, f"{bg_name}.jpg")

    subject.save(subject_path)
    background.save(background_path)

    subprocess.run([
        'python', '-m', 'demo.image_matting.colab.inference',
        '--input-path', INPUT_DIR,
        '--output-path', OUTPUT_DIR,
        '--ckpt-path', CKPT_PATH
    ], check=True, cwd='MODNet')

    matte_path = os.path.join(OUTPUT_DIR, 'subject.png')
    fg_rgba_path = os.path.join(OUTPUT_DIR, 'subject_fg.png')
    cleaned_path = os.path.join(OUTPUT_DIR, 'cleaned_fg.png')

    extract_subject(subject_path, matte_path, fg_rgba_path)
    remove_green_spill(fg_rgba_path, cleaned_path)

    bg_url = f"/static/bg/{bg_name}.jpg"
    fg_url = f"/static/fg/cleaned_fg.png"

    os.makedirs("static/bg", exist_ok=True)
    os.makedirs("static/fg", exist_ok=True)
    Image.open(background_path).convert('RGB').save(f"static/bg/{bg_name}.jpg")
    Image.open(cleaned_path).save("static/fg/cleaned_fg.png")

    position = load_position(bg_name) or {"x": 100, "y": 100, "scale": 1.0}
    return jsonify({
        "bg": bg_url,
        "fg": fg_url,
        "position": position,
        "bg_name": bg_name
    })

@app.route('/save_position', methods=['POST'])
def save_pos():
    data = request.json
    save_position(data['bg_name'], {
        "x": data['x'],
        "y": data['y'],
        "scale": data['scale']
    })
    return jsonify({"status": "success"})

@app.route('/composite', methods=['POST'])
def composite():
    data = request.json
    fg_path = 'static/fg/cleaned_fg.png'
    bg_path = f"static/bg/{data['bg_name']}.jpg"
    output_path = 'static/composited.png'
    composite_with_position(fg_path, bg_path, data['x'], data['y'], data['scale'], output_path)
    return jsonify({"url": "/static/composited.png"})
from flask import render_template

@app.route('/user')
def user_page():
    # List backgrounds for selection
    bg_files = os.listdir("static/bg")
    bg_names = [os.path.splitext(f)[0] for f in bg_files]
    return render_template("user.html", backgrounds=bg_names)

@app.route('/process_user', methods=['POST'])
def process_user():
    subject = request.files['subject']
    bg_name = request.form['bg_name']
    subject_path = os.path.join(INPUT_DIR, 'user_subject.jpg')
    subject.save(subject_path)

    # Run MODNet
    subprocess.run([
        'python', '-m', 'demo.image_matting.colab.inference',
        '--input-path', INPUT_DIR,
        '--output-path', OUTPUT_DIR,
        '--ckpt-path', CKPT_PATH
    ], check=True, cwd='MODNet')

    # Extract and clean
    matte_path = os.path.join(OUTPUT_DIR, 'user_subject.png')
    fg_rgba_path = os.path.join(OUTPUT_DIR, 'user_fg.png')
    cleaned_path = os.path.join(OUTPUT_DIR, 'user_cleaned.png')

    extract_subject(subject_path, matte_path, fg_rgba_path)
    remove_green_spill(fg_rgba_path, cleaned_path)

    # Get position
    position = load_position(bg_name)
    if not position:
        return jsonify({"error": "No position set for background"}), 400

    # Composite
    bg_path = f"static/bg/{bg_name}.jpg"
    result_path = "static/final_result.png"
    composite_with_position(cleaned_path, bg_path, position['x'], position['y'], position['scale'], result_path)

    return jsonify({"url": "/static/final_result.png"})

if __name__ == '__main__':
    app.run(debug=True)
