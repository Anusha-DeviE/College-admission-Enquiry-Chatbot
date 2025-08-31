import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline


csv_path = 'D:\mine\College-Chatbot\Chatbot_data.csv'
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='latin-1')

df = df.dropna()
df.columns = [col.strip().lower() for col in df.columns]
print("✅ Dataset Loaded. Sample:")
print(df.head())


#vector Embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')
questions = df['queries'].tolist()
corpus_embeddings = embedder.encode(questions, convert_to_numpy=True)


#Indexing
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)


#LLM
generator = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    tokenizer="tiiuae/falcon-7b-instruct",
    device=0
    if torch.cuda.is_available()
    else -1,
    max_new_tokens=200
)


generator.model.save_pretrained("./falcon-7b-instruct/")
generator.tokenizer.save_pretrained("./falcon-7b-instruct/")


def get_response(user_query, top_k=1):
    embedding = embedder.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(embedding, top_k)
    retrieved_answer = df.iloc[indices[0][0]]['answers']

    prompt = f"""You are a helpful assistant that answers college admission queries.
Relevant info: {retrieved_answer}
Question: {user_query}
Answer:"""

    generated = generator(prompt)[0]['generated_text']
    response = generated.split("Answer:")[-1].strip()
    return response

print("\n Chatbot is ready! (type 'exit' to quit):\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print(" Chatbot: Goodbye!")
        break
    reply = get_response(user_input)
    print(" Chatbot:", reply)
    
    
    
import gradio as gr

def chatbot_interface(user_query):
    return get_response(user_query)

gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="College Inquiry Chatbot",
    description="Ask anything about college admissions, fees, or programs!"
).launch()
