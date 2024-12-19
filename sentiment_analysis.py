
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
import pandas as pd
from datasets import Dataset

# Load the dataset
file_path = "./Merged_Walmart_Reviews.csv"
reviews_df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Check if the dataset contains necessary columns (modify as per dataset structure)
if "review" not in reviews_df.columns:
    raise ValueError("The dataset must contain a 'review' column for sentiment analysis.")

# Filter reviews for stress or anxiety
stress_keywords = ["stress", "anxiety", "relax", "calm", "relieve"]
filtered_df = reviews_df[reviews_df["review"].str.contains("|".join(stress_keywords), case=False, na=False)]

if filtered_df.empty:
    raise ValueError("No reviews found containing stress or anxiety-related keywords.")

# Convert filtered DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(filtered_df)

# Define RoBERTa model and tokenizer (use pre-trained model for sentiment analysis)
model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # A suitable sentiment model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Preprocessing function for tokenization
def preprocess_function(examples):
    return tokenizer(examples["review"], truncation=True, padding="max_length", max_length=512)

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Prepare the dataset for PyTorch DataLoader
tokenized_dataset = tokenized_dataset.with_format("torch")

# Create DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=32)

# Model Inference
model.eval()
predictions = []

with torch.no_grad():
    for batch in dataloader:
        inputs = {key: batch[key] for key in ["input_ids", "attention_mask"]}
        outputs = model(**inputs)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1)
        predictions.extend(batch_predictions.cpu().numpy())

# Map predictions to labels
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
mapped_predictions = [label_map[pred] for pred in predictions]

# Add predictions to the filtered dataset
filtered_df["Sentiment"] = mapped_predictions

# Save the results to a CSV file
output_file = "./Stress_Anxiety_Sentiment_Analysis.csv"
filtered_df.to_csv(output_file, index=False)

print(f"Sentiment analysis complete. Results saved to {output_file}.")
