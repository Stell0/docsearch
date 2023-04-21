from openai import Embedding
import pandas as pd
import json

# load docs from json file
with open("data.json") as f:
    docs = json.load(f)

# convert it to dataframe
df = pd.DataFrame(docs)
df["primary_key"] = df['title'] + df['section_title']

df["text"] = df.text.apply(lambda x: x.replace("\n", " "))

# sort by primary key
df = df.sort_values(by = "primary_key")

# remove duplicate
df = df.drop_duplicates(subset=['primary_key'])

def get_embedding(text, model="text-embedding-ada-002"):
    return Embedding.create(input = text, model = model)['data'][0]['embedding']

# get embedding of each text
df['embedding'] = df.text.apply(lambda x: get_embedding(x))

try:
	dfold = pd.read_csv('embedding.csv')
	dfnew = pd.concat([dfold, df], ignore_index=True, sort=False)
except:
	dfnew = df
        
# save embedding to csv
dfnew.to_csv('embedding.csv')
