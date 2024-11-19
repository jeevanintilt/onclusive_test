from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

app = FastAPI()

# Load the fine-tuned model and tokenizer
model_dir = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Define input and output schemas
class ClaimInput(BaseModel):
    claim: str
    main_text: str

class PredictionOutput(BaseModel):
    veracity: str
    confidence: float

# Mapping from label index to veracity class
LABEL_MAP = {
    0: "true",
    1: "false",
    2: "unproven",
    3: "mixture"
}

@app.post("/claim/v1/predict", response_model=PredictionOutput)
def predict_veracity(input: ClaimInput):
    """
    Predict the veracity of a claim based on input claim and main_text.
    """
    # Combine claim and main_text
    combined_text = input.claim + " " + input.main_text

    # Tokenize the input
    tokens = tokenizer(
        combined_text,
        truncation=True,
        padding="max_length",
        max_length=4096,
        return_tensors="pt"
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs).item()
        confidence = probs[0, pred_label].item()

    # Map label to class
    veracity = LABEL_MAP[pred_label]

    return PredictionOutput(veracity=veracity, confidence=confidence)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)