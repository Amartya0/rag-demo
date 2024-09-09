import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def vectorize_text(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True,
                       padding=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()


def retrieve_paragraphs(question, top_k=5):
    question_vector = vectorize_text(question)
    question_vector = np.expand_dims(question_vector, axis=0)
    # Retrieve more to ensure uniqueness
    distances, indices = index.search(question_vector, top_k * 2)
    unique_paragraphs = []
    seen = set()
    for i in indices[0]:
        if paragraphs[i] not in seen:
            seen.add(paragraphs[i])
            unique_paragraphs.append(paragraphs[i])
        if len(unique_paragraphs) == top_k:
            break
    return unique_paragraphs


df = pd.read_csv('paragraphs.csv')
questions = df['Questions'].tolist()
paragraphs = df['Paragraphs'].tolist()


model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


index_file = 'paragraph_index.faiss'
if os.path.exists(index_file):
    print("Loading existing index...")
    index = faiss.read_index(index_file)
else:
    print("Creating new index...")
    concatenated_texts = [q + " " + p for q, p in zip(questions, paragraphs)]
    vectorized_texts = [vectorize_text(text) for text in concatenated_texts]

    vectorized_texts = np.array(vectorized_texts)
    embedding_dim = vectorized_texts.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(vectorized_texts)
    faiss.write_index(index, index_file)


question = "Why is understanding contaminant transport and fate crucial?"
retrieved_paragraphs = retrieve_paragraphs(question, 3)
for i, paragraph in enumerate(retrieved_paragraphs):
    print(f"Paragraph {i + 1}: {paragraph}", end='\n\n')
