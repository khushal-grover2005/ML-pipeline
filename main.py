import os, json, torch, cv2, numpy as np, time, threading, base64, io
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import nest_asyncio
from pyngrok import ngrok, conf
from fastapi import FastAPI, File, UploadFile, Header
from fastapi.responses import JSONResponse
from PIL import Image
import requests
import uvicorn
import sys
import platform
import subprocess
import socket
import urllib3

# Suppress only the single warning from urllib3 needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Import our configuration
from config import settings, validate_setup

# Global variable for class_map
class_map = None

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from the given port"""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return start_port  # Return original if none found

def setup_model():
    """Load and setup the ML model"""
    global class_map
    
    print("📁 Checking existing files:")
    print(f"Model: {'✅' if os.path.exists(settings.model_path) else '❌'} {settings.model_path}")
    print(f"Classes: {'✅' if os.path.exists(settings.class_path) else '❌'} {settings.class_path}")
    
    # Validate setup
    if not validate_setup():
        sys.exit(1)
    
    # Load class mapping
    with open(settings.class_path) as f:
        class_map = {int(k): v for k, v in json.load(f).items()}
    
    num_classes = len(class_map)
    print(f"📊 Loaded {num_classes} classes")
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(settings.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, device

def setup_gradcam(model, device):
    """Setup Grad-CAM functionality"""
    target_layers = [model.features[-1]]
    
    def preprocess_bgr_for_model(img_bgr, img_size=224):
        img = cv2.resize(img_bgr, (img_size, img_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
        return img_rgb, tensor
    
    def predict_with_gradcam(img_bgr, topk=3):
        rgb_normed, tensor = preprocess_bgr_for_model(img_bgr, 224)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # Fixed: get first element
        
        topk_idx = probs.argsort()[-topk:][::-1]
        
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=tensor, 
                              targets=[ClassifierOutputTarget(int(topk_idx[0]))])[0]  # Fixed: get first element
        
        cam_vis = show_cam_on_image(rgb_normed, grayscale_cam, use_rgb=True)
        
        # Fixed: Proper indexing
        topk_pairs = []
        for i in topk_idx:
            topk_pairs.append({
                "label": class_map[int(i)], 
                "prob": float(probs[int(i)])
            })
        
        return topk_pairs, cam_vis
    
    print("✅ Grad-CAM function ready!")
    return predict_with_gradcam

def setup_fastapi(predict_function):
    """Setup FastAPI application"""
    app = FastAPI(
        title="Plant Disease Inference API",
        version=settings.model_version,
        description="AI-powered plant disease detection with Grad-CAM visualization"
    )
    
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": "mobilenet_v2",
            "version": settings.model_version,
            "port": settings.port,
            "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        }
    
    @app.post("/infer")
    async def infer(file: UploadFile = File(...), authorization: str = Header(default="")):
        if authorization != f"Bearer {settings.api_token}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            topk, cam_vis = predict_function(img_bgr, topk=3)
            top = topk[0]  # Fixed: get first element of topk
            
            cam_bgr = cv2.cvtColor(cam_vis, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", cam_bgr)
            heatmap_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            
            return {
                "disease": top["label"],
                "confidence": top["prob"],
                "topk": topk,
                "heatmap_b64": heatmap_b64,
                "modelVersion": settings.model_version
            }
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    return app

def kill_ngrok_processes():
    """Kill any existing ngrok processes"""
    try:
        if platform.system() == "Windows":
            os.system("taskkill /f /im ngrok.exe >nul 2>&1")
        else:
            os.system("pkill -f ngrok >/dev/null 2>&1")
        time.sleep(2)
    except:
        pass

def kill_process_on_port(port):
    """Kill process using the specified port on Windows"""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["netstat", "-ano", "|", "findstr", f":{port}"],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if f":{port}" in line:
                        parts = line.split()
                        pid = parts[-1]
                        subprocess.run(["taskkill", "/f", "/pid", pid], 
                                     capture_output=True)
                        print(f"✅ Killed process {pid} on port {port}")
                        time.sleep(2)
                        break
    except:
        pass

def setup_ngrok():
    """Setup ngrok tunnel using local ngrok.exe"""
    if not settings.use_ngrok:
        print("ℹ️  ngrok is disabled in settings")
        return None
    
    print("🔧 Setting up ngrok tunnel...")
    
    try:
        # Kill any existing ngrok processes
        kill_ngrok_processes()
        
        # Add our local ngrok directory to PATH temporarily
        ngrok_dir = os.path.dirname(os.path.abspath(settings.ngrok_path))
        original_path = os.environ.get('PATH', '')
        os.environ['PATH'] = ngrok_dir + os.pathsep + original_path
        
        # Set auth token
        conf.get_default().auth_token = settings.ngrok_token
        
        # Test if ngrok executable works
        try:
            # Test ngrok version command
            result = subprocess.run(
                [settings.ngrok_path, "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise Exception(f"ngrok test failed: {result.stderr}")
        except Exception as e:
            print(f"❌ ngrok executable test failed: {e}")
            return setup_ngrok_manual_fallback()
        
        # Connect using pyngrok
        try:
            public_url = ngrok.connect(addr=str(settings.port), proto="http", bind_tls=True)
            print(f"✅ Public URL: {public_url.public_url}")
            return public_url
        except Exception as e:
            print(f"❌ pyngrok connect failed: {e}")
            return setup_ngrok_manual_fallback()
            
    except Exception as e:
        print(f"❌ ngrok setup failed: {e}")
        print("🔧 Continuing with local-only access")
        return None

def setup_ngrok_manual_fallback():
    """Fallback method to start ngrok manually"""
    print("🔄 Trying manual ngrok startup...")
    
    try:
        # Kill any existing ngrok
        kill_ngrok_processes()
        
        # Start ngrok tunnel manually
        cmd = [
            settings.ngrok_path,
            "http",
            "--authtoken", settings.ngrok_token,
            str(settings.port)
        ]
        
        # Start ngrok in background
        if platform.system() == "Windows":
            # On Windows, use CREATE_NO_WINDOW to hide console window
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            # On Unix-like systems
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Wait for ngrok to start
        time.sleep(5)
        
        # Try to get the public URL from ngrok's API
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=10, verify=False)
            if response.status_code == 200:
                tunnels = response.json().get('tunnels', [])
                if tunnels:
                    public_url = tunnels[0]['public_url']
                    print(f"✅ Public URL (manual): {public_url}")
                    # Create a simple object to mimic pyngrok's response
                    return type('NgrokTunnel', (), {'public_url': public_url})()
        except requests.RequestException:
            print("⚠️  Could not get ngrok URL from API")
            print("💡 Check https://dashboard.ngrok.com/ for active tunnels")
        
        return None
        
    except Exception as e:
        print(f"❌ Manual ngrok setup failed: {e}")
        return None

def test_api(public_url=None):
    """Test the API endpoints"""
    print("🧪 Running API tests...")
    
    # Test localhost first (more reliable)
    base_url = f"http://localhost:{settings.port}"
    headers = {"Authorization": f"Bearer {settings.api_token}"}
    
    # Wait a bit longer to ensure server is fully ready
    time.sleep(3)
    
    try:
        # Health check - test multiple times if needed
        health_success = False
        for attempt in range(3):
            try:
                health_response = requests.get(f"{base_url}/health", timeout=10)
                print(f"🩺 Health check attempt {attempt + 1}: {health_response.status_code}")
                
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    print(f"✅ Health: {health_data}")
                    health_success = True
                    break
                else:
                    print(f"⚠️ Health check failed: {health_response.text}")
                    time.sleep(2)  # Wait before retrying
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Health check error (attempt {attempt + 1}): {e}")
                time.sleep(2)
        
        if not health_success:
            print("❌ Health check failed after multiple attempts")
            return False
        
        # Create a proper test image (not random)
        # Create a simple green image that looks like a plant leaf
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        test_img[:, :, 1] = 100  # Green channel
        test_img[50:150, 50:150, 1] = 200  # Brighter green square in center
        
        # Add some noise to make it more realistic
        noise = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
        test_img = np.clip(test_img + noise, 0, 255)
        
        # Encode as JPEG
        _, buf = cv2.imencode(".jpg", test_img)
        image_data = buf.tobytes()
        
        # Test inference
        files = {"file": ("test_leaf.jpg", image_data, "image/jpeg")}
        
        print("📤 Testing inference endpoint...")
        response = requests.post(f"{base_url}/infer", files=files, headers=headers, timeout=30)
        print(f"📊 Inference status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ API Test SUCCESS!")
                print(f"🏷️  Disease: {result['disease']}")
                print(f"📈 Confidence: {result['confidence']:.4f}")
                
                # Show top predictions
                if 'topk' in result:
                    print("🔝 Top predictions:")
                    for i, pred in enumerate(result['topk'][:3], 1):
                        print(f"   {i}. {pred['label']} ({pred['prob']:.4f})")
                
                print(f"\n🎯 MERN Integration URLs:")
                if public_url:
                    print(f"📍 Public: {public_url.public_url}/infer")
                print(f"📍 Local: http://localhost:{settings.port}/infer")
                print(f"🔑 Auth: Bearer {settings.api_token}")
                
                return True
                
            except requests.exceptions.JSONDecodeError:
                print(f"❌ Inference returned invalid JSON: {response.text}")
                return False
                
        else:
            print(f"❌ Test Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main application entry point"""
    print(f"🚀 Starting Plant Disease Detection API v{settings.model_version}")
    
    # Setup components
    model, device = setup_model()
    predict_function = setup_gradcam(model, device)
    app = setup_fastapi(predict_function)
    
    # Setup ngrok if enabled
    public_url = setup_ngrok()
    
    # Start server in background thread
    def start_server():
        nest_asyncio.apply()
        uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(5)
    print(f"🔗 Server running on http://localhost:{settings.port}")
    
    # Test the API with retry logic
    test_success = False
    for attempt in range(3):
        print(f"🧪 Test attempt {attempt + 1}/3")
        test_success = test_api(public_url)
        if test_success:
            break
        time.sleep(3)
    
    if not test_success:
        print("⚠️  API tests failed, but server may still be running")
        print(f"🔗 Local URL: http://localhost:{settings.port}")
        if public_url:
            print(f"🌐 Public URL: {public_url.public_url}")
    
    # Keep server running
    try:
        print("\n📡 Server is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if public_url:
            try:
                ngrok.disconnect(public_url.public_url)
            except:
                pass
        # Kill ngrok process
        kill_ngrok_processes()

if __name__ == "__main__":
    main()