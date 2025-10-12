import os
import cohere
import guardrails as gd
from guardrails.hub import ValidRange, ValidChoices
from pydantic import BaseModel, Field
from rich import print
from typing import List
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import evaluate

# Load environment variables from .env file
load_dotenv()

# Setup the Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY")) # Get your free API key: https://dashboard.cohere.com/api-keys

reference="Because the sound quality is the best out there"
generated1="Because the audio experience is unrivaled"
generated2="Because the microphone has the best quality"

# Load BERTScore metric from Hugging Face
bertscore = evaluate.load("bertscore")

# Calculate BERTScore for generated1
results1 = bertscore.compute(predictions=[generated1], references=[reference], lang="en")
print("\n=== BERTScore for Generated 1 ===")
print(f"Precision: {results1['precision'][0]:.4f}")
print(f"Recall: {results1['recall'][0]:.4f}")
print(f"F1 Score: {results1['f1'][0]:.4f}")

# Calculate BERTScore for generated2
results2 = bertscore.compute(predictions=[generated2], references=[reference], lang="en")
print("\n=== BERTScore for Generated 2 ===")
print(f"Precision: {results2['precision'][0]:.4f}")
print(f"Recall: {results2['recall'][0]:.4f}")
print(f"F1 Score: {results2['f1'][0]:.4f}")

# Initialize ROUGE scorer with unigram (rouge1)
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

# Calculate ROUGE-1 for generated1
rouge1_scores1 = scorer.score(reference, generated1)
print("\n=== ROUGE-1 (Unigram) for Generated 1 ===")
print(f"Precision: {rouge1_scores1['rouge1'].precision:.4f}")
print(f"Recall: {rouge1_scores1['rouge1'].recall:.4f}")
print(f"F1 Score: {rouge1_scores1['rouge1'].fmeasure:.4f}")

# Calculate ROUGE-1 for generated2
rouge1_scores2 = scorer.score(reference, generated2)
print("\n=== ROUGE-1 (Unigram) for Generated 2 ===")
print(f"Precision: {rouge1_scores2['rouge1'].precision:.4f}")
print(f"Recall: {rouge1_scores2['rouge1'].recall:.4f}")
print(f"F1 Score: {rouge1_scores2['rouge1'].fmeasure:.4f}")

