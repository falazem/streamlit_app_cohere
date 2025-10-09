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
