import os
import torch
import time
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from index_builder import load_data, load_model
from retriever import retrieve_relevant_paragraphs, load_metadata, sanity_check
import faiss
import re
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("generation_pipeline.log"), logging.StreamHandler()])


def load_generation_model(model_name, device):
    """ Load the generation model (e.g., BART, T5) """
    logging.info(f"Loading generation model: {model_name} onto {device}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    logging.info(
        f"Generation model loaded in {time.time() - start_time:.2f} seconds")
    return tokenizer, model


def is_valid_sentence(sentence):
    """ Validate that a sentence starts with uppercase and ends with punctuation """
    return sentence[0].isupper() and re.match(r'.*[.!?]$', sentence)


def generate_answer(question, paragraphs, gen_tokenizer, gen_model, device):
    """ Generate answer based on the question and retrieved paragraphs """
    logging.info("Generating an answer...")

    input_text = f"question: {question} context: {' '.join(paragraphs)}"
    logging.info(
        f"Concatenated question and context. Total length: {len(input_text)} characters")

    # Tokenize and generate
    inputs = gen_tokenizer(input_text, return_tensors="pt",
                           truncation=True, padding=True, max_length=512).to(device)

    start_time = time.time()
    outputs = gen_model.generate(
        **inputs,
        max_length=512,
        num_beams=5,  # Change to 7 for singular generation
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        early_stopping=True
    )
    generation_time = time.time() - start_time
    logging.info(f"Answer generated in {generation_time:.2f} seconds")

    # Decode the generated answer
    raw_answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info("Answer decoding completed")

    # Split the generated text into sentences
    sentences = sent_tokenize(raw_answer)
    valid_sentences = [
        sentence for sentence in sentences if is_valid_sentence(sentence)]

    return ' '.join(valid_sentences)


def main():
    logging.info("Starting the answer generation program...")

    data_file = os.path.join(
        'data', 'test_Paragraphs_Questions_Answers.csv')
    index_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_index.faiss')
    metadata_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_index_metadata.json')

    if not os.path.exists(index_file):
        logging.error(
            f"Index file not found at {index_file}. Please run the index_builder script first.")
        raise FileNotFoundError(
            f"Index file not found at {index_file}. Please run the index_builder script first.")

    logging.info("Loading data and metadata...")
    questions, paragraphs = load_data(data_file)
    metadata = load_metadata(metadata_file)

    retrieval_model_name = 'sentence-transformers/all-MiniLM-L12-v2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tokenizer, model = load_model(retrieval_model_name, device)
    sanity_check(metadata, retrieval_model_name,
                 retrieval_model_name, paragraphs)

    logging.info(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)

    generation_model_name = 'facebook/bart-large'
    gen_tokenizer, gen_model = load_generation_model(
        generation_model_name, device)

    query = "Why is the heat loss calculation often done by month?"
    logging.info(f"Retrieving relevant paragraphs for query: '{query}'")

    start_time = time.time()
    relevant_paragraphs = retrieve_relevant_paragraphs(
        query, index, tokenizer, model, paragraphs, device)
    retrieval_time = time.time() - start_time
    logging.info(
        f"Paragraph retrieval completed in {retrieval_time:.2f} seconds")

    start_time = time.time()
    answer = generate_answer(query, relevant_paragraphs,
                             gen_tokenizer, gen_model, device)
    generation_time = time.time() - start_time
    logging.info(
        f"Answer generation completed in {generation_time:.2f} seconds")

    logging.info("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()
