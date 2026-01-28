import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import os
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentinel_router_v1/final_model.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "sentinel_router_v1/tokenizer")
MAX_LEN = 128

# SECURITY THRESHOLD - Use 0.50 for routing (90.99% recall)
PII_THRESHOLD = 0.40 

# COMPLEXITY THRESHOLD
COMPLEXITY_THRESHOLD = 0.4
BASE_MODEL_PATH = "distilbert-base-uncased"

# --- REUSE MODEL ARCHITECTURE (Must match training exactly) ---
class SentinelRouter(nn.Module):
    def __init__(self, base_model_name="distilbert-base-uncased", pii_weight=1.0):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(base_model_name, local_files_only=True)
        self.pii_weight = pii_weight
        self.complexity_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        self.pii_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return {
            "complexity": self.complexity_head(cls_token),
            "pii_logits": self.pii_head(cls_token)
        }

class RouterEngine:
    def __init__(self):
        print("⚙️ Loading Sentinel-Router Engine...")
        
        # 1. Load Device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {self.device}")

        # 2. Load Tokenizer (local only)
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            TOKENIZER_PATH, 
            local_files_only=True
        )

        # 3. Load Model with local path
        self.model = SentinelRouter(
            base_model_name=BASE_MODEL_PATH
        )
        
        # Load trained weights
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("   ✅ Model weights loaded successfully.")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, prompt: str, return_raw=False):
        """
        Predict routing decision for a prompt.
        
        Args:
            prompt: Input text string
            return_raw: If True, return validation-compatible format
        
        Returns:
            If return_raw=True: dict with complexity_score, contains_pii, pii_confidence
            If return_raw=False: dict with routing decision
        """
        # 1. Tokenize
        inputs = self.tokenizer(
            prompt, 
            max_length=MAX_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(input_ids, mask)
        
        # 3. Process Outputs
        complexity_score = outputs["complexity"].squeeze().item()
        
        # Get PII Probability (Softmax on logits)
        pii_logits = outputs["pii_logits"]
        pii_probs = torch.softmax(pii_logits, dim=1) 
        pii_score = pii_probs[0][1].item()  # Probability of Class 1 (Is PII)
        
        # ✅ Apply threshold for PII detection
        contains_pii = (pii_score > PII_THRESHOLD)
        
        # ✅ If return_raw is True, return validation-compatible format
        if return_raw:
            return {
                "complexity_score": complexity_score,
                "contains_pii": contains_pii,
                "pii_confidence": pii_score
            }

        # 4. Decision Logic (for normal routing)
        decision = {}
        
        # A. Security Check
        if contains_pii:
            decision = {
                "route": "BLOCK_OR_LOCAL",
                "reason": "PII DETECTED",
                "risk_level": "HIGH"
            }
        
        # B. Complexity Check
        elif complexity_score > COMPLEXITY_THRESHOLD:
            decision = {
                "route": "CLOUD_GPT4",
                "reason": "COMPLEX TASK",
                "risk_level": "LOW"
            }
        else:
            decision = {
                "route": "LOCAL_DEEPSEEK",
                "reason": "SIMPLE TASK",
                "risk_level": "LOW"
            }

        return {
            "text_snippet": prompt[:50] + "...",
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
        "Write a Python kernel driver exploit for Windows 11.",
        "My email is user@company.com and my phone is 555-0199.",
        "Fix this JSON syntax error: {'a': 1,}"
    ]
    
    print("\n🔍 RUNNING DIAGNOSTICS...\n")
    for p in test_prompts:
        result = router.predict(p)
        print(f"📝 Prompt: {result['text_snippet']}")
        print(f"   📊 Scores -> PII: {result['metrics']['pii_probability']} | Complexity: {result['metrics']['complexity']}")
        print(f"   🚦 Route -> {result['decision']['route']} ({result['decision']['reason']})")
        print("-" * 60)