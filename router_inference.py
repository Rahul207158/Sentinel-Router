import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import json

# ============================================================
# MODEL ARCHITECTURE (Same as training)
# ============================================================

class SentinelRouter(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(base_model_name)
        
        # Complexity head
        self.complexity_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # PII head
        self.pii_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        complexity_score = self.complexity_head(cls_token)
        pii_logits = self.pii_head(cls_token)
        
        return complexity_score, pii_logits


# ============================================================
# INFERENCE CLASS
# ============================================================

class RouterInference:
    def __init__(self, model_dir="./sentinel_router_v1", device=None):
        """
        Initialize the router inference engine.
        
        Args:
            model_dir: Directory containing the trained model and tokenizer
            device: torch device (cuda/cpu). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading model on {self.device}...")
        
        # Load tokenizer
        tokenizer_path = f"{model_dir}/tokenizer"
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        
        # Initialize model architecture
        self.model = SentinelRouter("distilbert-base-uncased")
        
        # Load trained weights
        model_path = f"{model_dir}/final_model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def predict(self, text, return_raw=False):
        """
        Make routing decision for a single text prompt.
        
        Args:
            text: Input prompt string
            return_raw: If True, return raw scores. If False, return routing decision.
        
        Returns:
            dict with keys:
                - complexity_score: float (0.0-1.0)
                - contains_pii: bool
                - pii_confidence: float (0.0-1.0)
                - suggested_route: str ("local-deepseek" | "cloud-gpt4" | "block-pii")
                - complexity_tier: str ("low" | "medium" | "high")
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Inference
        with torch.no_grad():
            complexity_score, pii_logits = self.model(input_ids, attention_mask)
        
        # Process outputs
        complexity_score = complexity_score.item()
        pii_probs = torch.softmax(pii_logits, dim=1)
        pii_confidence = pii_probs[0, 1].item()  # Probability of PII class
        contains_pii = pii_confidence > 0.5
        
        if return_raw:
            return {
                "complexity_score": complexity_score,
                "pii_confidence": pii_confidence,
                "contains_pii": contains_pii
            }
        
        # Determine complexity tier
        if complexity_score < 0.33:
            complexity_tier = "low"
        elif complexity_score < 0.67:
            complexity_tier = "medium"
        else:
            complexity_tier = "high"
        
        # Routing logic
        if contains_pii:
            suggested_route = "block-pii"
        elif complexity_tier == "high":
            suggested_route = "cloud-gpt4"
        else:
            suggested_route = "local-deepseek"
        
        return {
            "text": text,
            "complexity_score": round(complexity_score, 3),
            "complexity_tier": complexity_tier,
            "contains_pii": contains_pii,
            "pii_confidence": round(pii_confidence, 3),
            "suggested_route": suggested_route
        }
    
    def predict_batch(self, texts, batch_size=32):
        """
        Make routing decisions for multiple prompts efficiently.
        
        Args:
            texts: List of input prompt strings
            batch_size: Number of prompts to process at once
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.predict(text))
        return results


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Initialize router
    router = RouterInference()
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "My email is john.doe@example.com and my SSN is 123-45-6789",
        "Explain quantum entanglement and its applications in quantum computing",
        "Convert this text to uppercase",
        "Design a distributed rate limiter using Redis with sliding window algorithm"
    ]
    
    print("\n" + "="*80)
    print("🔍 ROUTING DECISIONS")
    print("="*80 + "\n")
    
    for prompt in test_prompts:
        result = router.predict(prompt)
        
        # Color coding for route
        route_emoji = {
            "local-deepseek": "🟢",
            "cloud-gpt4": "🔵",
            "block-pii": "🔴"
        }
        
        print(f"{route_emoji.get(result['suggested_route'], '⚪')} {result['suggested_route'].upper()}")
        print(f"   Text: {result['text'][:70]}{'...' if len(result['text']) > 70 else ''}")
        print(f"   Complexity: {result['complexity_tier']} ({result['complexity_score']:.3f})")
        print(f"   PII: {'⚠️ Detected' if result['contains_pii'] else '✓ Safe'} ({result['pii_confidence']:.3f})")
        print()
