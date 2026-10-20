"""
ApexSR - Peak UI SRGAN Super-Resolution Web Application Server
Powered by PyTorch & TensorLayerX SRGAN Generator
"""
import os
import io
import time
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Ensure TensorLayerX uses the PyTorch backend
if 'TL_BACKEND' not in os.environ:
    os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
from srgan import SRGAN_g

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, 'static')
SAMPLES_DIR = os.path.join(STATIC_DIR, 'samples')
WEIGHTS_PATH = os.path.join(APP_DIR, 'models', 'g_torch.npz')
if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH = os.path.join(APP_DIR, 'models', 'g.npz')

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# Global model instance
G_MODEL = None
MODEL_LOADED = False
LOAD_ERROR = None

def init_model():
    global G_MODEL, MODEL_LOADED, LOAD_ERROR
    try:
        print(f"[*] Initializing SRGAN Generator (Backend: {tlx.BACKEND})...")
        g = SRGAN_g()
        g.init_build(tlx.nn.Input(shape=(1, 3, 96, 96)))
        print(f"[*] Loading weights from: {WEIGHTS_PATH}")
        if WEIGHTS_PATH.endswith('g_torch.npz') or tlx.BACKEND == 'torch':
            g.load_weights(WEIGHTS_PATH)
        else:
            g.load_weights(WEIGHTS_PATH, format='npz_dict')
        g.set_eval()
        
        # Warmup forward pass
        dummy = np.zeros((1, 3, 96, 96), dtype=np.float32)
        dummy_tensor = tlx.ops.convert_to_tensor(dummy)
        _ = g(dummy_tensor)
        
        G_MODEL = g
        MODEL_LOADED = True
        print("[+] SRGAN Generator successfully initialized and warmed up!")
    except Exception as e:
        LOAD_ERROR = str(e)
        print(f"[!] Model initialization error: {e}")

def calculate_sharpness(img):
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10((255.0 ** 2) / mse))

def img_to_base64(img, fmt='png', quality=95):
    """Encode OpenCV BGR image to base64 Data URL."""
    if fmt.lower() in ['jpg', 'jpeg']:
        ext = '.jpg'
        mime = 'image/jpeg'
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif fmt.lower() == 'webp':
        ext = '.webp'
        mime = 'image/webp'
        params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    else:
        ext = '.png'
        mime = 'image/png'
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
    
    success, buffer = cv2.imencode(ext, img, params)
    if not success:
        raise ValueError("Failed to encode image to buffer")
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:{mime};base64,{b64_str}"

def apply_post_processing(img, sharpness=0.0, denoise=0.0, contrast_boost=False):
    """Optional post-refinement enhancements."""
    result = img.copy()
    
    # 1. Edge-preserving Denoising (bilateral filter)
    if denoise > 0.05:
        sigma = int(10 + denoise * 30)
        result = cv2.bilateralFilter(result, d=5, sigmaColor=sigma, sigmaSpace=sigma)
        
    # 2. Contrast & Dynamic Range Optimization (CLAHE in LAB space)
    if contrast_boost:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    # 3. Unsharp Mask Sharpening
    if sharpness > 0.05:
        amount = float(sharpness) * 0.8
        gaussian = cv2.GaussianBlur(result, (0, 0), sigmaX=1.5)
        result = cv2.addWeighted(result, 1.0 + amount, gaussian, -amount, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
    return result

def run_srgan_inference(img_bgr, max_dim=640, sharpness=0.2, denoise=0.0, contrast_boost=False, fmt='png'):
    """
    Super-resolve an input image by 4x using SRGAN.
    """
    if G_MODEL is None:
        raise RuntimeError("SRGAN Model is not loaded or failed to initialize.")
        
    t_start = time.time()
    orig_h, orig_w = img_bgr.shape[:2]
    
    # Adaptive resizing if input exceeds max_dim to maintain fast interactive response
    if max_dim > 0 and (orig_w > max_dim or orig_h > max_dim):
        scale = max_dim / float(max(orig_w, orig_h))
        new_w = max(16, int(orig_w * scale))
        new_h = max(16, int(orig_h * scale))
        lr_img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        lr_img = img_bgr.copy()
        
    cur_h, cur_w = lr_img.shape[:2]
    
    # Convert OpenCV BGR to RGB for neural network
    lr_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    
    # Prepare normalized tensor [-1, 1] in NCHW layout
    lr_tensor = (lr_rgb / 127.5) - 1.0
    lr_tensor = np.asarray(lr_tensor, dtype=np.float32)
    lr_tensor = np.transpose(lr_tensor, axes=[2, 0, 1])
    lr_tensor = lr_tensor[np.newaxis, :, :, :]
    lr_tensor = tlx.ops.convert_to_tensor(lr_tensor)
    
    # Forward pass
    t_inf_start = time.time()
    sr_tensor = G_MODEL(lr_tensor)
    t_inf_end = time.time()
    inference_ms = (t_inf_end - t_inf_start) * 1000.0
    
    # Convert back from [-1, 1] NCHW to uint8 HWC RGB, then to OpenCV BGR
    sr_np = tlx.ops.convert_to_numpy(sr_tensor)
    sr_np = np.transpose(sr_np[0], axes=[1, 2, 0])
    sr_rgb = np.clip((sr_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    sr_img = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
    
    # Baseline comparison: Standard Bicubic 4x
    bicubic_img = cv2.resize(lr_img, (cur_w * 4, cur_h * 4), interpolation=cv2.INTER_CUBIC)
    
    # Apply post-processing if configured
    refined_sr_img = apply_post_processing(sr_img, sharpness=sharpness, denoise=denoise, contrast_boost=contrast_boost)
    
    # Metrics
    psnr_score = calculate_psnr(bicubic_img, refined_sr_img)
    lr_sharpness = calculate_sharpness(lr_img)
    sr_sharpness = calculate_sharpness(refined_sr_img)
    sharpness_gain = ((sr_sharpness - lr_sharpness) / max(lr_sharpness, 1.0)) * 100.0
    
    t_end = time.time()
    total_ms = (t_end - t_start) * 1000.0
    
    out_h, out_w = refined_sr_img.shape[:2]
    
    return {
        'input_b64': img_to_base64(lr_img, fmt=fmt),
        'srgan_b64': img_to_base64(refined_sr_img, fmt=fmt),
        'bicubic_b64': img_to_base64(bicubic_img, fmt=fmt),
        'metrics': {
            'input_width': cur_w,
            'input_height': cur_h,
            'input_megapixels': round((cur_w * cur_h) / 1000000.0, 3),
            'output_width': out_w,
            'output_height': out_h,
            'output_megapixels': round((out_w * out_h) / 1000000.0, 3),
            'scale_factor': '4x',
            'pixel_multiplier': '16x',
            'inference_time_ms': round(inference_ms, 1),
            'total_time_ms': round(total_ms, 1),
            'psnr_vs_bicubic': round(psnr_score, 2),
            'lr_sharpness': round(lr_sharpness, 1),
            'sr_sharpness': round(sr_sharpness, 1),
            'sharpness_gain_pct': round(sharpness_gain, 1)
        }
    }

# Ensure static directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(STATIC_DIR, path)

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'online',
        'model_loaded': MODEL_LOADED,
        'backend': str(tlx.BACKEND),
        'weights_file': os.path.basename(WEIGHTS_PATH),
        'architecture': 'SRGAN Generator (16 Residual Blocks + Sub-pixel Conv)',
        'scale': '4x Super-Resolution',
        'error': LOAD_ERROR
    })

@app.route('/api/samples', methods=['GET'])
def get_samples():
    """List preset sample images available for 1-click testing."""
    sample_items = [
        {
            'id': 'landscape_river',
            'title': 'Scenic Valley & River',
            'category': 'Landscape',
            'description': 'Fine water ripples, mountain foliage, and aerial textures.',
            'filename': 'valid_lr.png'
        },
        {
            'id': 'portrait_face',
            'title': 'High Detail Portrait',
            'category': 'Portrait',
            'description': 'Facial micro-details, hair texture, and skin gradient fidelity.',
            'filename': 'input_lr_demo.png'
        },
        {
            'id': 'cyber_fashion',
            'title': 'Studio Model Shoot',
            'category': 'Fashion',
            'description': 'Fabric weave, clothing edges, and lighting contrast.',
            'filename': 'model_crop_lr.png'
        },
        {
            'id': 'macro_eye',
            'title': 'Macro Iris & Texture',
            'category': 'Macro',
            'description': 'Intricate iris fibers and high-frequency reflection patterns.',
            'filename': 'macro_lr.png'
        }
    ]
    
    # Return samples with check if file exists
    available = []
    for s in sample_items:
        full_path = os.path.join(SAMPLES_DIR, s['filename'])
        if os.path.exists(full_path):
            s['url'] = f"/samples/{s['filename']}"
            available.append(s)
            
    return jsonify({'samples': available})

@app.route('/api/refine', methods=['POST'])
def refine_image():
    try:
        # Check inputs
        img_bgr = None
        
        # 1. Uploaded file
        if 'image' in request.files:
            file = request.files['image']
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
        # 2. JSON Base64
        elif request.is_json and 'image_base64' in request.json:
            b64_data = request.json['image_base64']
            if ',' in b64_data:
                b64_data = b64_data.split(',', 1)[1]
            decoded = base64.b64decode(b64_data)
            data = np.frombuffer(decoded, dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
        # 3. Sample ID
        elif (request.is_json and 'sample_id' in request.json) or ('sample_id' in request.form):
            sample_id = request.json.get('sample_id') if request.is_json else request.form.get('sample_id')
            sample_map = {
                'landscape_river': 'valid_lr.png',
                'portrait_face': 'input_lr_demo.png',
                'cyber_fashion': 'model_crop_lr.png',
                'macro_eye': 'macro_lr.png'
            }
            fname = sample_map.get(sample_id)
            if fname:
                full_path = os.path.join(SAMPLES_DIR, fname)
                if os.path.exists(full_path):
                    img_bgr = cv2.imread(full_path)
                    
        if img_bgr is None:
            return jsonify({'error': 'No valid image received. Please upload an image or choose a sample.'}), 400
            
        # Options
        data_source = request.json if request.is_json else request.form
        max_dim = int(data_source.get('max_dim', 640))
        sharpness = float(data_source.get('sharpness', 0.25))
        denoise = float(data_source.get('denoise', 0.0))
        contrast_boost = str(data_source.get('contrast_boost', 'false')).lower() in ['true', '1', 'yes']
        output_format = str(data_source.get('output_format', 'png')).lower()
        
        result = run_srgan_inference(
            img_bgr,
            max_dim=max_dim,
            sharpness=sharpness,
            denoise=denoise,
            contrast_boost=contrast_boost,
            fmt=output_format
        )
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_model()
    port = int(os.environ.get('PORT', 5050))
    print(f"\n🚀 ApexSR Studio running at: http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
