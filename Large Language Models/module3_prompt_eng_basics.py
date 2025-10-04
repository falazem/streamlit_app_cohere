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

def generate_text(message):
    # Generate the response by streaming it
    response = co.chat_stream(model="command-a-03-2025",
                   messages=[{'role':'user', 'content': message}])

    for event in response:
        if event.type == "content-delta":
            print(event.delta.message.content.text, end="")

generate_text("Generate a concise product description for the product: wireless earbuds.")

generate_text("""
    Generate a concise product description for the product: wireless earbuds.
    Use the following format: Hook, Solution, Features and Benefits, Call to Action.
    """)

generate_text("""
    Summarize this email in one sentence.
    Dear [Team Members],
    I am writing to thank you for your hard work and dedication in organizing our recent community meetup. The event was a great success and it would not have been possible without your efforts.
    I am especially grateful for the time and energy you have invested in making this event a reality. Your commitment to ensuring that everything ran smoothly and that our guests had a great time is greatly appreciated.
    I am also thankful for the support and guidance you have provided to me throughout the planning process. Your insights and ideas have been invaluable in ensuring that the event was a success.
    I am confident that our community will benefit greatly from this event and I am excited to see the positive impact it will have.
    Thank you again for your hard work and dedication. I am looking forward to working with you on future events.
    Sincerely,
    [Your Name]
    """)

generate_text("""
    Extract the movie title from the text below.
    Deadpool 2 | Official HD Deadpool's "Wet on Wet" Teaser | 2018
    """)

generate_text("""
    Given the following text, write down a list of potential frequently asked questions (FAQ), together with the answers.
    The Cohere Platform provides an API for developers and organizations to access cutting-edge LLMs without needing machine learning know-how.
    The platform handles all the complexities of curating massive amounts of text data, model development, distributed training, model serving, and more.
    This means that developers can focus on creating value on the applied side rather than spending time and effort on the capability-building side.

    There are two key types of language processing capabilities that the Cohere Platform provides — text generation and text embedding — and each is served by a different type of model.

    With text generation, we enter a piece of text, or prompt, and get back a stream of text as a completion to the prompt.
    One example is asking the model to write a haiku (the prompt) and getting an originally written haiku in return (the completion).

    With text embedding, we enter a piece of text and get back a list of numbers that represents its semantic meaning (we’ll see what “semantic” means in a section below).
    This is useful for use cases that involve “measuring” what a passage of text represents, for example, in analyzing its sentiment.
    """)