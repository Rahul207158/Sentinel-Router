import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer

# --- CONFIGURATION ---
MODEL_PATH = "./sentinel_router_v1/final_model.pt"
OUTPUT_ONNX_PATH = "./sentinel_router_v1/sentinel.onnx"
MAX_LEN = 128

# --- RE-DEFINE MODEL (Must match training exactly) ---
class SentinelRouter(nn.Module):
    def __init__(self, base_model_name="distilbert-base-uncased"):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(base_model_name)
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
        return self.complexity_head(cls_token), self.pii_head(cls_token)

def export():
    print("⚙️ Loading PyTorch Model...")
    device = torch.device("cpu") # Export on CPU is safer
    model = SentinelRouter()
    
    # Load Weights
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        state_dict = torch.load(MODEL_PATH, map_location=device)
    else:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
    model.load_state_dict(state_dict)
    model.eval()

    # Create Dummy Input (The model needs to trace a fake execution)
    print("🧪 Creating Dummy Input...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    text = "This is a sample input to trace the graph."
    inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, padding="max_length", truncation=True)
    
    dummy_input = (inputs["input_ids"], inputs["attention_mask"])

    print(f"📦 Exporting to {OUTPUT_ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX_PATH,
        export_params=True,
        opset_version=14, # Standard for Transformers
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['complexity_score', 'pii_logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'complexity_score': {0: 'batch_size'},
            'pii_logits': {0: 'batch_size'}
        }
    )
    print("✅ ONNX Export Complete!")

if __name__ == "__main__":
    export()