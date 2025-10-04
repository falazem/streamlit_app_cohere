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
message = "I have been a Senior Manager for an AI team at bell canada for 4 years. I have been constantly performing. What advice do you give me to get promotted to Director?"

# Create a custom system message
system_message="""## Task and Context
You are an assistant who provides advice to employees on how to get promoted in their companies. You have been working in HR for 20 years and have helped hundreds of employees get promoted. You are very good at understanding the context of the employee and providing tailored advice.
"""

# Add the messages
messages = [{'role': 'system', 'content': system_message},
            {'role': 'user', 'content': message}]

#Generate the response
response = co.chat(model="command-a-03-2025",
                messages=messages)


# response = co.chat_stream(model="command-a-03-2025",
#                           messages=[{'role':'user', 'content': message}])

# for event in response:
#     if event.type == "content-delta":
#         print(event.delta.message.content.text, end="")                  

print(response.message.content[0].text)

# Append the previous response
messages.append({'role': 'assistant', 'content': response.message.content[0].text})

# Add the user message
message = "Summarize your advice in 3 main bullet points."

# Append the user message
messages.append({'role': 'user', 'content': message})

# Generate the response with the current chat history as the context
response = co.chat(model="command-a-03-2025",
                   messages=messages)

print(response.message.content[0].text)

# Append the previous response
messages.append({'role': 'assistant', 'content': response.message.content[0].text})

# Add the user message
message = "Thanks. I am of Middle Eastern origin with a First Name that is difficult to pronounce. I see very little ethnic diversity in the executive front. What should I do so that I don't become a victim of unconscious bias?"

# Append the user message
messages.append({'role': 'user', 'content': message})

# Generate the response with the current chat history as the context
response = co.chat(model="command-a-03-2025",
                   messages=messages)

print(response.message.content[0].text)

# Append the previous response
messages.append({'role': 'assistant', 'content': response.message.content[0].text})
# View the chat history
for message in messages:
    print(message,"\n")