import os
import onnxruntime as ort
import numpy as np
from transformers import DistilBertTokenizer
import re

# --- CONFIGURATION ---
MODEL_PATH = "./sentinel_router_v1/sentinel.onnx"
TOKENIZER_PATH = "./sentinel_router_v1/tokenizer"
MAX_LEN = 128

# --- THRESHOLDS ---
# Locked in from Phase 2 validation
PII_THRESHOLD = 0.3
COMPLEXITY_THRESHOLD = 0.4

# --- REGEX SAFETY NET ---
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
SSN_REGEX = r"\b\d{3}-\d{2}-\d{4}\b"

class RouterEngine:
    def __init__(self):
        print("⚡ Loading Sentinel-Router (ONNX Optimized)...")
        
        # 1. Load Tokenizer
        # We still use the lightweight tokenizer from transformers
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
        except:
            print("   ⚠️ Local tokenizer not found, downloading default...")
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # 2. Load ONNX Session
        # This replaces the massive PyTorch model loading
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"❌ ONNX model not found at {MODEL_PATH}. Did you run export_onnx.py?")
             
        # Load optimized runtime
        self.session = ort.InferenceSession(MODEL_PATH)
        print("   ✅ ONNX Runtime Initialized.")

    def _softmax(self, x):
        """Compute softmax values for each set of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(self, prompt: str):
        # 1. Tokenize (CPU side)
        inputs = self.tokenizer(
            prompt, 
            max_length=MAX_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="np" # Return NumPy arrays, not Pytorch tensors
        )
        
        # Prepare inputs for ONNX
        # Names 'input_ids' and 'attention_mask' must match export script
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }

        # 2. Inference (The fast part)
        # Returns list: [complexity_score, pii_logits]
        ort_outs = self.session.run(None, ort_inputs)
        
        # 3. Post-Processing (Numpy Math)
        
        # Output 0: Complexity (Already Sigmoid-ed in model)
        # Shape is (1, 1), we need a float
        complexity_score = float(ort_outs[0][0][0]) 
        
        # Output 1: PII Logits (Raw scores)
        # Shape is (1, 2) -> [Safe_Score, PII_Score]
        pii_logits = ort_outs[1][0]
        
        # Apply Softmax to get probability
        pii_probs = self._softmax(pii_logits)
        pii_score = float(pii_probs[1]) # Probability of class 1 (PII)

        # 4. Logic & Decision
        decision = {}
        
        # A. Neural PII Check
        if pii_score > PII_THRESHOLD:
            decision = {
                "route": "BLOCK_OR_LOCAL",
                "reason": f"NEURAL PII DETECTED ({pii_score:.2f})",
                "risk_level": "HIGH"
            }
        
        # B. Regex Backup
        elif re.search(EMAIL_REGEX, prompt) or re.search(SSN_REGEX, prompt):
            decision = {
                "route": "BLOCK_OR_LOCAL",
                "reason": "REGEX PATTERN DETECTED",
                "risk_level": "HIGH"
            }
            
        # C. Complexity Check
        elif complexity_score > COMPLEXITY_THRESHOLD:
            decision = {
                "route": "CLOUD_GPT4",
                "reason": f"COMPLEX TASK ({complexity_score:.2f})",
                "risk_level": "LOW"
            }
        else:
            decision = {
                "route": "LOCAL_DEEPSEEK",
                "reason": f"SIMPLE TASK ({complexity_score:.2f})",
                "risk_level": "LOW"
            }

        return {
            "metrics": {
                "complexity": round(complexity_score, 4),
                "pii_probability": round(pii_score, 4)
            },
            "decision": decision
        }

# --- TEST DRIVE ---
if __name__ == "__main__":
    router = RouterEngine()
    
    test_prompts = [
        "What is the capital of France?",
        "Write a Python kernel driver exploit.",
        "Update record for user john@example.com",
        "Explain quantum entanglement."
    ]
    
    print("\n🏎️  RUNNING ONNX SPEED TEST...\n")
    import time
    
    for p in test_prompts:
        start = time.time()
        result = router.predict(p)
        latency = (time.time() - start) * 1000 # ms
        
        print(f"📝 '{p[:30]}...'")
        print(f"   ⏱️  Latency: {latency:.2f}ms")
        print(f"   📊 PII: {result['metrics']['pii_probability']} | Complexity: {result['metrics']['complexity']}")
        print(f"   🚦 Route: {result['decision']['route']}")
        print("-" * 50)