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

#define documents in a list

documents = [
    {
        "title": "Tall penguins",
        "text": "Emperor penguins are the tallest."},
    {
        "title": "Penguin habitats",
        "text": "Emperor penguins only live in Antarctica."},
    {
        "title": "What are animals?",
        "text": "Animals are different from plants."}
]

# Get the user message
message = "What are the tallest living penguins?"

# Generate the response
response = co.chat_stream(message=message,
                          documents=documents)

# Display the response
citations = []
cited_documents = []

for event in response:
    if event.event_type == "text-generation":
        print(event.text, end="")
    elif event.event_type == "citation-generation":
        citations.extend(event.citations)
    elif event.event_type == "stream-end":
      cited_documents = event.response.documents

# Display the citations and source documents
if citations:
  print("\n\nCITATIONS:")
  for citation in citations:
    print(citation)

  print("\nDOCUMENTS:")
  for document in cited_documents:
    print(document)