import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import json
import numpy as np

df = pd.read_csv('embedding.csv')

# drop index column
df = df.drop(columns = ['Unnamed: 0'])

df['embedding'] = df.embedding.apply(eval).apply(np.array)

while True:
	# get search query from user
	query = input("inserisci la tua domanda: ")

	# get embedding of query
	query_embedding = get_embedding(query, engine='text-embedding-ada-002')

	# calculate cosine similarity
	df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

	# sort by similarity
	res = df.sort_values(by = "similarity", ascending = False).head(10)

	res=res.drop_duplicates(subset=['text'])
	
	# get a list of urls
	urls = res.url.tolist()

	# put all the text results in a string
	kb = ""
	for i in range(len(res)):
		kb += res.iloc[i].text
	
	prompt = '''Given the above knoledge base, answer the following question in the same language it's written.
	question: "{query}"'''
	prompt = prompt.format(query = query)

	# take only the first (2k token * 4) 8000 - prompt characters of the knowledge base.
	kb = kb[:8000 - len(prompt)]

	# Send a request to generate response using the GPT API
	completion = openai.Completion.create(
    	engine='text-davinci-003',
    	prompt=prompt,
    	temperature=0.5,
    	max_tokens=2000
	)

	print(completion['choices'][0]['text'] + "\n##############################################\n")

	for i in range(len(res)):
		print(res.iloc[i].url)
