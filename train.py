import os
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics like accuracy and F1-score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

def evaluate_model(trainer, test_dataset, label_map):
    """
    Evaluate the model on the test dataset and print classification report.
    """
    predictions = trainer.predict(test_dataset)
    y_true = test_dataset["labels"]
    y_pred = np.argmax(predictions.predictions, axis=-1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_map.values()))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def train_model(data_dir: str, output_dir: str, epochs: int, batch_size: int, lr: float):
    """
    Train the model and evaluate its performance.
    """
    print("Loading dataset...")
    dataset = load_from_disk(data_dir)

    print("Loading model and tokenizer...")
    model_name = "nbroad/longformer-base-health-fact"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving the fine-tuned model to {output_dir}...")
    trainer.save_model(output_dir)

    print("Evaluating the model...")
    label_map = {0: "true", 1: "false", 2: "unproven", 3: "mixture"}
    evaluate_model(trainer, dataset["test"], label_map)
    print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Longformer Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./processed_data",
        help="Directory containing the processed dataset (default: ./processed_data)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine_tuned_model",
        help="Directory to save the fine-tuned model (default: ./fine_tuned_model)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)"
    )
    args = parser.parse_args()

    train_model(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr)