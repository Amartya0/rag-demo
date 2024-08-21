import torch
import numpy as np
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import faiss
from fuzzywuzzy import fuzz

# Load DPR models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base")

# Load BART model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Sample corpus (reference answers)
corpus = [
    "The capital of France is Paris.",
    "The process of photosynthesis involves the conversion of light energy into chemical energy.",
    "In 1492, Christopher Columbus sailed the ocean blue."
]

# Encode corpus
encoded_corpus = []
for doc in corpus:
    inputs = context_tokenizer(doc, return_tensors="pt")
    outputs = context_encoder(**inputs)
    encoded_corpus.append(outputs.pooler_output.detach().cpu().numpy())

# Convert to faiss index
index = faiss.IndexFlatIP(768)
index.add(np.vstack(encoded_corpus))

# Function to retrieve relevant document


def retrieve_document(question, top_k=1):
    inputs = question_tokenizer(question, return_tensors="pt")
    outputs = question_encoder(**inputs)
    question_embedding = outputs.pooler_output.detach().cpu().numpy()
    D, I = index.search(question_embedding, top_k)
    return [corpus[i] for i in I[0]]

# Function to generate response based on retrieved document


def generate_response(retrieved_doc, question):
    inputs = bart_tokenizer(retrieved_doc + " " + question,
                            return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs["input_ids"], num_beams=4, min_length=30, max_length=100, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Example question and student's answer
question = "What is the capital of France?"
student_answer = "Paris is the capital of France."

# Retrieve relevant document
retrieved_doc = retrieve_document(question)[0]
print("Retrieved Document:", retrieved_doc)

# Generate response
generated_response = generate_response(retrieved_doc, question)
print("Generated Response:", generated_response)

# Score using fuzzy matching
similarity = fuzz.token_set_ratio(student_answer, generated_response)
print("Similarity:", similarity)

# Threshold for determining correctness
threshold = 80
score = int(similarity >= threshold)
print("Score:", score)
