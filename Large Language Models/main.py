import json
import os
import textwrap
import cohere
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Annotated
from pydantic import Field
from dotenv import load_dotenv

