import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  
import cohere
import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

load_dotenv()

co = cohere.ClientV2(os.getenv("API_KEY")) 
texts = ["I like to be in my house", 
         "I enjoy staying home", 
         "the isotope 238u decays to 206pb"]

response = co.embed(
    texts=texts,
    model='embed-v4.0',
    input_type='search_document',
    embedding_types=['float']
)

embeddings = response.embeddings.float
[sentence1, sentence2, sentence3] = embeddings

# print("Embedding for sentence 1", np.array(sentence1))
# print("Embedding for sentence 2", np.array(sentence2))
# print("Embedding for sentence 3", np.array(sentence3))

#calculate the dot product

# print("Similarity between sentences 1 and 2:", np.dot(sentence1, sentence2))
# print("Similarity between sentences 1 and 3:", np.dot(sentence1, sentence3))
# print("Similarity between sentences 2 and 3:", np.dot(sentence2, sentence3))

#cosine similarity
print("Cosine similarity between sentences 1 and 2:", cosine_similarity([sentence1], [sentence2])[0][0])  
print("Cosine similarity between sentences 1 and 3:", cosine_similarity([sentence1], [sentence3])[0][0])  
print("Cosine similarity between sentences 2 and 3:", cosine_similarity([sentence2], [sentence3])[0][0])