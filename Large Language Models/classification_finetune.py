import os
import json
import numpy as np
import pandas as pd
import cohere
from cohere import ClassifyExample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from dotenv import load_dotenv
load_dotenv()

co = cohere.ClientV2(os.getenv("API_KEY")) 

# Load the dataset to a dataframe
df = pd.read_csv('https://raw.githubusercontent.com/cohere-ai/notebooks/main/notebooks/data/atis_subset.csv', names=['query','intent'])
print(df.head())
# unique_intents = df['intent'].unique()
# print(unique_intents)

# Split the dataset into training and test portions
df_train, df_test = train_test_split(df, test_size=200, random_state=21)

def create_classification_data(text, label):
    formatted_data = {
        "text": text,
        "label": label
    }
    return formatted_data

if not os.path.isfile("data.jsonl"):
    print("Creating jsonl file ...")
    with open("data.jsonl", 'w+') as file:
        for row in df_train.itertuples():
            formatted_data = create_classification_data(row.query, row.intent)
            file.write(json.dumps(formatted_data) + '\n')
        file.close()
        print("Done")
else:
    print("data.jsonl file already exists")