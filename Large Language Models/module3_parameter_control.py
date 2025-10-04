import pandas as pd
import numpy as np
import altair as alt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv
import cohere


load_dotenv()
co = cohere.ClientV2(os.getenv("API_KEY")) 

# Add the user message
message = "Hello."

# Generate the response
response = co.chat(model="command-a-03-2025",
                   messages=[{'role':'user', 'content': message}])

print(response.message.content[0].text)

# Add the user message
message = "I like learning about the industrial revolution and how it shapes the modern world. How can I introduce myself in two words."

# Generate the response multiple times by specifying a low temperature value
for idx in range(3):
    response = co.chat(model="command-a-03-2025",
                       messages=[{'role':'user', 'content': message}],
                       temperature=0)

    print(f"{idx+1}: {response.message.content[0].text}\n")


# Add the user message
message = "I like learning about the industrial revolution and how it shapes the modern world. How can I introduce myself in two words."

# Generate the response multiple times by specifying a high temperature value
for idx in range(3):
    response = co.chat(model="command-a-03-2025",
                       messages=[{'role':'user', 'content': message}],
                       temperature=1)

    print(f"{idx+1}: {response.message.content[0].text}\n")
