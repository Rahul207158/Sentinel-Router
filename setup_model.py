import os
import urllib.request
import sys

# --- CONFIGURATION ---
REPO_URL = "https://github.com/Rahul207158/Sentinel-Router/releases/download/v1.0.0/sentinel.onnx"
DEST_FOLDER = "./sentinel_router_v1"
DEST_FILE = f"{DEST_FOLDER}/sentinel.onnx"

def setup():
    # 1. Create Folder
    if not os.path.exists(DEST_FOLDER):
        print(f"📂 Creating directory: {DEST_FOLDER}")
        os.makedirs(DEST_FOLDER)

    # 2. Check if model exists
    if os.path.exists(DEST_FILE):
        print("✅ Model already exists. Skipping download.")
        return

    # 3. Download Model
    print(f"⬇️  Downloading Sentinel V1 Model from GitHub Releases...")
    print(f"   Source: {REPO_URL}")
    
    try:
        def progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r   Progress: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(REPO_URL, DEST_FILE, progress)
        print("\n✅ Download Complete!")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("   Please manually download 'sentinel.onnx' from the GitHub Releases page.")

if __name__ == "__main__":
    setup()