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