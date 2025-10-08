import json
import os
import textwrap
import cohere
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Annotated
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup the Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY")) # Get your free API key: https://dashboard.cohere.com/api-keys

def generate_text(prompt, temp=0):
  response = co.chat_stream(
    message=prompt,
    model="command-a-03-2025",
    temperature=temp)

  for event in response:
      if event.event_type == "text-generation":
          print(event.text, end='')

user_input_product = "a wireless headphone product named the CO-1T"
user_input_keywords = '"bluetooth", "wireless", "fast charging"'
user_input_customer = "a software developer who works in noisy offices"
user_input_describe = "benefits of this product"

prompt = f"""Write a creative product description for {user_input_product}.
Keywords: {user_input_keywords}
Audience: {user_input_customer}
Describe: {user_input_describe}"""

generate_text(prompt, temp=0.5)     