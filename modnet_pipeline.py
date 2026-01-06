import os
import subprocess
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import streamlit.components.v1 as components
import json
import base64
from streamlit_js_eval import streamlit_js_eval


# === CONFIGURATION === #
BASE_DIR = os.path.abspath('modnet_workspace')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
BG_DIR = os.path.join(BASE_DIR, 'background')
CKPT_PATH = os.path.abspath(os.path.join('MODNet', 'pretrained', 'modnet_photographic_portrait_matting.ckpt'))
POSITION_FILE = 'foreground_position.json'


# === HELPER FUNCTIONS === #
def setup_workspace():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BG_DIR, exist_ok=True)


def clone_modnet_repo():
    if not os.path.exists('MODNet'):
        subprocess.run(['git', 'clone', 'https://github.com/ZHKKKe/MODNet'], check=True)


def run_modnet_inference():
    subprocess.run([
        'python', '-m', 'demo.image_matting.colab.inference',
        '--input-path', INPUT_DIR,
        '--output-path', OUTPUT_DIR,
        '--ckpt-path', CKPT_PATH
    ], check=True, cwd='MODNet')


def extract_subject(image_path, matte_path, save_path):
    img = np.array(Image.open(image_path).convert('RGB'))
    matte = np.array(Image.open(matte_path).convert('L')) / 255.0
    matte = np.stack([matte] * 3, axis=-1)
    fg = (img * matte).astype(np.uint8)
    alpha = (matte[..., 0] * 255).astype(np.uint8)
    fg_rgba = np.dstack((fg, alpha))
    Image.fromarray(fg_rgba).save(save_path)
    return save_path


def remove_green_spill(image_path, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    hsv[:, :, 1] = np.where(mask > 0, hsv[:, :, 1] * 0.5, hsv[:, :, 1])
    cleaned = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if alpha is not None:
        cleaned = cv2.merge((cleaned, alpha))
    cv2.imwrite(save_path, cleaned)
    return save_path


def image_to_base64(path):
    with open(path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def composite_with_position(fg_path, bg_path, x, y, scale, save_path):
    fg = Image.open(fg_path).convert("RGBA")
    bg = Image.open(bg_path).convert("RGBA")

    # Retain aspect ratio and quality for the background
    bg_width, bg_height = bg.size
    canvas_width, canvas_height = bg_width, bg_height

    fg_scaled = fg.resize((int(fg.width * scale), int(fg.height * scale)))
    result = bg.copy()
    result.paste(fg_scaled, (int(x), int(y)), fg_scaled)
    result.save(save_path)
    return save_path


# === JSON FILE HANDLING === #
def load_position(background_path):
    """Load saved position for a specific background."""
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, 'r') as f:
            positions = json.load(f)
        return positions.get(background_path, None)
    return None


def save_position(foreground_path, position_data):
    """Save position for a specific foreground."""
    positions = {}
    if (os.path.exists(POSITION_FILE)):
        with open(POSITION_FILE, 'r') as f:
            positions = json.load(f)
    positions[foreground_path] = position_data
    with open(POSITION_FILE, 'w') as f:
        json.dump(positions, f)


# === UI === #
st.set_page_config(page_title="MODNet Editor", layout="wide")
st.title("MODNet Bg Replacement")

# === IMAGE UPLOAD === #
st.header("Upload Images")

subject_file = st.file_uploader("Upload Subject Image", type=['jpg', 'jpeg', 'png'])
background_file = st.file_uploader("Upload Background Image", type=['jpg', 'jpeg', 'png'])
bg_name = st.text_input("Enter Background Name (for saving)")

if subject_file and background_file and bg_name:
    with st.spinner("Processing with MODNet..."):
        setup_workspace()
        clone_modnet_repo()

        subject_path = os.path.join(INPUT_DIR, 'subject.jpg')
        background_path = os.path.join(BG_DIR, f"{bg_name}.jpg")
        with open(subject_path, 'wb') as f:
            f.write(subject_file.read())
        with open(background_path, 'wb') as f:
            f.write(background_file.read())

        st.success(f"Background '{bg_name}' saved successfully!")

        run_modnet_inference()

        matte_path = os.path.join(OUTPUT_DIR, 'subject.png')
        transparent_fg = extract_subject(subject_path, matte_path, os.path.join(OUTPUT_DIR, 'subject_fg.png'))
        cleaned_fg = remove_green_spill(transparent_fg, os.path.join(OUTPUT_DIR, 'cleaned_fg.png'))

        # Get dimensions of the background image
        bg_image = Image.open(background_path)
        bg_width, bg_height = bg_image.size

        # Load saved position for the current background
        saved_position = load_position(background_path)
        if saved_position:
            fgX, fgY, scale = saved_position['x'], saved_position['y'], saved_position['scale']
        else:
            fgX, fgY, scale = 100, 100, 1  # Default position and scale

        # Base64 images for use in the JS canvas
        bg_base64 = image_to_base64(background_path)
        fg_base64 = image_to_base64(cleaned_fg)

        # HTML + JavaScript for canvas editor
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <body style="margin:0;">

        <!-- Scrollable container -->
        <div style="width: 100%; height: 650px; overflow: auto; border: 1px solid #ccc;">
            <canvas id="canvas" width="{bg_width}" height="{bg_height}"></canvas>
        </div>

        <!-- Display Coordinates -->
        <div style="position: fixed; top: 10px; left: 10px; color: #fff; background-color: rgba(0, 0, 0, 0.5); padding: 5px;">
            Coordinates: <span id="coordinates">X: {fgX}, Y: {fgY}, Scale: {scale}</span>
        </div>

        <!-- Download Button -->
        <div style="position: fixed; top: 50px; left: 10px;">
            <button id="downloadBtn" style="padding: 10px; background-color: #007bff; color: white; border: none; cursor: pointer;">
                Download Image
            </button>
        </div>

        <script>
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');

        let bg = new Image();
        let fg = new Image();

        bg.src = "data:image/jpeg;base64,{bg_base64}";
        fg.src = "data:image/png;base64,{fg_base64}";

        let fgX = {fgX}, fgY = {fgY}, scale = {scale};
        let dragging = false;
        let offsetX = 0, offsetY = 0;

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);
            ctx.drawImage(fg, fgX, fgY, fg.width * scale, fg.height * scale);

            // Update coordinates display
            document.getElementById('coordinates').innerText = 'X: ' + fgX.toFixed(2) + ', Y: ' + fgY.toFixed(2) + ', Scale: ' + scale.toFixed(2);

            // Send updated coordinates to Streamlit
            window.parent.postMessage({{ x: fgX, y: fgY, scale: scale }}, "*");
        }}

        canvas.onmousedown = function(e) {{
            offsetX = e.offsetX - fgX;
            offsetY = e.offsetY - fgY;
            dragging = true;
        }};

        canvas.onmouseup = function() {{
            dragging = false;
        }};

        canvas.onmousemove = function(e) {{
            if (dragging) {{
                fgX = e.offsetX - offsetX;
                fgY = e.offsetY - offsetY;
                draw();
            }}
        }};

        canvas.onwheel = function(e) {{
            e.preventDefault();
            scale += e.deltaY * -0.001;
            scale = Math.min(Math.max(0.1, scale), 5);
            draw();
        }};

        bg.onload = fg.onload = function() {{
            draw();
        }};

        // Download functionality
        document.getElementById('downloadBtn').onclick = function() {{
            const link = document.createElement('a');
            link.download = 'composited_image.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }};
        </script>

        </body>
        </html>
        """

        # Render the editor
        components.html(html_code, height=650)

        # Handle incoming coordinates from JavaScript
        if "position_data" not in st.session_state:
            st.session_state["position_data"] = {"x": fgX, "y": fgY, "scale": scale}

        # JavaScript to Streamlit communication
        st.markdown(
            """
            <script>
            window.addEventListener("message", (event) => {
                const positionData = event.data;
                if (positionData && positionData.x !== undefined) {
                    // Update Streamlit session state with the new position data
                    const streamlitEvent = new CustomEvent("streamlit:setComponentValue", {
                        detail: positionData
                    });
                    window.dispatchEvent(streamlitEvent);
                    // Update Streamlit session state
                    fetch("/_stcore/streamlit/session_state", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(positionData)
                    });
                }
            });
            </script>
            """,
            unsafe_allow_html=True,
        )

        # Save Position handling on the server
        # Add logging to track the save process
        if st.button("Save Position"):
            try:
                # Log the current session state position data
                st.write("Saving position data:", st.session_state.get("position_data", {}))

                # Save the updated position of the foreground (subject)
                position_data = st.session_state["position_data"]
                save_position(background_path, position_data)  # Save using the background path

                st.success("Foreground position saved successfully!")
            except Exception as e:
                st.error(f"Failed to save foreground position: {e}")
                st.write("Error details:", e)

        # Update JavaScript to ensure Streamlit session state is updated
        st.markdown(
            """
            <script>
            window.addEventListener("message", (event) => {
                const positionData = event.data;
                if (positionData && positionData.x !== undefined) {
                    // Log the received position data
                    console.log("Received position data from canvas:", positionData);

                    // Update Streamlit session state with the new position data
                    fetch("/_stcore/streamlit/session_state", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ position_data: positionData })
                    }).then(() => {
                        console.log("Position data sent to Streamlit session state:", positionData);
                    }).catch((error) => {
                        console.error("Failed to send position data to Streamlit session state:", error);
                    });
                }
            });
            </script>
            """,
            unsafe_allow_html=True,
        )
