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
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"),log_warning_experimental_features=False) # Get your free API key: https://dashboard.cohere.com/api-keys

def classify_sentiment(product_review):
        # Create prompt with examples
        prompt = """Classify this text into positive, negative, or neutral sentiment. Here are some examples:

        Positive examples:
        - "The order came 5 days early"
        - "The item exceeded my expectations" 
        - "I ordered more for my friends"
        - "I would buy this again"
        - "I would recommend this to others"

        Negative examples:
        - "The package was damaged"
        - "The order is 5 days late"
        - "The order was incorrect" 
        - "I want to return my item"
        - "The item's material feels low quality"

        Neutral examples:
        - "The item was nothing special"
        - "I would not buy this again but it wasn't a waste of money"
        - "The item was neither amazing or terrible"
        - "The item was okay"
        - "I have no emotions towards this item"

        Text to classify:
        {}"""

        res = co.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(product_review)
                }
            ],
            temperature=0.0,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "class": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"]
                        }
                    },
                    "required": ["class"]
                }
            }
        )
        return json.loads(res.message.content[0].text)["class"]


app = FastAPI()

class ProductReviews(BaseModel):
    reviews: Annotated[List[str], Field(min_length=1)]

@app.post("/prediction")
def predict_sentiment(product_reviews: ProductReviews):
    sentiments = []
    for review in product_reviews.reviews:
        sentiments.append(classify_sentiment(review))
    return sentiments

#To enable the server on our local host

# uvicorn main:app --reload
#curl command

# curl 'http://127.0.0.1:9002/prediction' \
#   -H 'Accept-Language: en-US,en;q=0.9' \
#   -H 'Connection: keep-alive' \
#   -H 'Content-Type: application/json' \
#   -H 'Origin: http://127.0.0.1:9002' \
#   -H 'Referer: http://127.0.0.1:9002/docs' \
#   -H 'Sec-Fetch-Dest: empty' \
#   -H 'Sec-Fetch-Mode: cors' \
#   -H 'Sec-Fetch-Site: same-origin' \
#   -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36' \
#   -H 'accept: application/json' \
#   -H 'sec-ch-ua: "Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"' \
#   -H 'sec-ch-ua-mobile: ?0' \
#   -H 'sec-ch-ua-platform: "macOS"' \
#   --data-raw $'{\n  "reviews": [\n    "Terrible food but great hotel\u0021","cute cat","beautiful view","Love the club but bad coaching"\n  ]\n}'