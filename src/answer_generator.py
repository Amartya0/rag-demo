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
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_generation_model(model_name, device):
    """ Load the generation model (e.g., BART, T5) """
    logging.info(f"Loading generation model: {model_name} onto {device}")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    logging.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return tokenizer, model


def is_valid_sentence(sentence):
    """ Validate that a sentence starts with uppercase and ends with punctuation """
    return sentence[0].isupper() and re.match(r'.*[.!?]$', sentence)


def generate_answer(question, paragraphs, gen_tokenizer, gen_model, device):
    """ Generate answer based on the question and retrieved paragraphs """
    # logging.info("Generating an answer...")

    # Concatenate the question and retrieved paragraphs for input
    input_text = f"question: {question} context: {' '.join(paragraphs)}"

    # Tokenize and generate
    inputs = gen_tokenizer(input_text, return_tensors="pt",
                           truncation=True, padding=True, max_length=512).to(device)

    outputs = gen_model.generate(
        **inputs,
        max_length=512,  # Increase the max_length to allow a more detailed answer
        num_beams=7,     # Use beam search for better quality
        no_repeat_ngram_size=3,  # Prevent repetitive phrases
        length_penalty=2.0,  # Encourage the model to generate longer sequences
        early_stopping=False  # Ensure the model generates until it reaches max_length or EOS
    )

    # Decode the generated answer
    raw_answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split the generated text into sentences
    sentences = sent_tokenize(raw_answer)

    # Filter the sentences using our validation function
    valid_sentences = [
        sentence for sentence in sentences if is_valid_sentence(sentence)]

    # logging.info("Answer generation completed.")
    return ' '.join(valid_sentences)


def main():
    """ Main function to run retrieval and generation pipeline """
    logging.info("Starting the program...")

    # Define file paths
    data_file = os.path.join(
        'data', 'test_Paragraphs_Questions_Answers_Grades.csv')
    index_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_Grades_index.faiss')
    metadata_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_Grades_index_metadata.json')

    if not os.path.exists(index_file):
        logging.error(
            f"Index file not found at {index_file}. Please run the index_builder script first.")
        raise FileNotFoundError(
            f"Index file not found at {index_file}. Please run the index_builder script first.")

    # Load data (questions and paragraphs) and metadata
    logging.info("Loading data and metadata...")
    questions, paragraphs = load_data(data_file)
    metadata = load_metadata(metadata_file)

    # Load retrieval model (e.g., sentence transformers)
    retrieval_model_name = 'sentence-transformers/all-MiniLM-L12-v2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading retrieval model: {retrieval_model_name}")
    tokenizer, model = load_model(retrieval_model_name, device)

    # Perform sanity check on the metadata
    sanity_check(metadata, retrieval_model_name,
                 retrieval_model_name, paragraphs)

    # Load FAISS index for paragraph retrieval
    logging.info(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)

    # Load the generation model (e.g., BART, T5)
    generation_model_name = 'facebook/bart-large'
    gen_tokenizer, gen_model = load_generation_model(
        generation_model_name, device)

    # Example query
    query = "When did humanity's ecological footprint surpass Earth's capacity?"

    # Retrieve relevant paragraphs with progress visualization
    logging.info(f"Retrieving relevant paragraphs for the query: '{query}'")
    relevant_paragraphs = retrieve_relevant_paragraphs(
        query, index, tokenizer, model, paragraphs, device)

    # Generate answer using the retrieved paragraphs
    answer = generate_answer(query, relevant_paragraphs,
                             gen_tokenizer, gen_model, device)

    # Display the generated answer
    logging.info("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()
