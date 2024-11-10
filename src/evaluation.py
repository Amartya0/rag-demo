import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from retriever import retrieve_relevant_paragraphs
from index_builder import load_data, load_model, vectorize_text
from answer_generator import load_generation_model, generate_answer
import faiss
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("evaluation_pipeline.log"), logging.StreamHandler()])


def evaluate_with_bleu(generated_answer, test_answer):
    """ Evaluate the BLEU score between the generated answer (reference) and the test answer """
    reference = [generated_answer.split(
    )]  # Reference needs to be tokenized and in a list
    candidate = test_answer.split()  # Candidate (test answer) also tokenized
    smoothing = SmoothingFunction().method4
    score = sentence_bleu(reference, candidate, weights=(
        1, 0, 0, 0), smoothing_function=smoothing)
    logging.info(f"Computed BLEU Score: {score}")
    return score


def evaluate_with_semantic_similarity(generated_answer, test_answer, model, tokenizer, device):
    """ Evaluate the semantic similarity between the generated answer and the test answer """
    inputs = tokenizer([generated_answer, test_answer], return_tensors="pt",
                       padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    similarity = F.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
    logging.info(f"Computed Semantic Similarity: {similarity}")
    return similarity


def compute_weighted_score(bleu_score, semantic_similarity):
    """ Compute a weighted score out of 10 with 60% weight on BLEU and 40% weight on Semantic Similarity """

    weighted_score = (0.4 * bleu_score + 0.6 * semantic_similarity) * 10
    logging.info(f"Computed Weighted Score: {weighted_score}")
    return weighted_score


def main():
    """ Main function to run retrieval, generation, and evaluation pipeline """
    logging.info("Starting the evaluation process...")

    # Define file paths
    data_file = os.path.join(
        'data', 'test_Paragraphs_Questions_Answers.csv')
    index_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_index.faiss')
    metadata_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_index_metadata.json')

    # Check if necessary files exist
    if not os.path.exists(data_file):
        logging.error(f"Data file not found at {data_file}.")
        return
    if not os.path.exists(index_file):
        logging.error(
            f"Index file not found at {index_file}. Please run the index builder script first.")
        return

    # Load data (questions, paragraphs, answers, grades)
    logging.info("Loading data...")
    questions, paragraphs = load_data(data_file)

    retrieval_model_name = 'sentence-transformers/all-MiniLM-L12-v2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(
        f"Loading retrieval model: {retrieval_model_name} on {device}")
    tokenizer, model = load_model(retrieval_model_name, device)

    # Load FAISS index
    logging.info(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)

    # Load generation model (e.g., BART, T5)
    generation_model_name = 'facebook/bart-large'
    gen_tokenizer, gen_model = load_generation_model(
        generation_model_name, device)

    # Example query for evaluation
    query = "Why is the heat loss calculation often done by month?"
    test_answer = "The heat loss calculation is often done by month because the heat loss is typically highest in December and January."
    logging.info(f"Evaluating query: '{query}'")
    logging.info(f"Test Answer: '{test_answer}'")

    # Retrieve relevant paragraphs for the query
    relevant_paragraphs = retrieve_relevant_paragraphs(
        query, index, tokenizer, model, paragraphs, device)

    # Generate an answer based on the retrieved paragraphs
    generated_answer = generate_answer(
        query, relevant_paragraphs, gen_tokenizer, gen_model, device)
    logging.info(f"Generated Answer (used as reference): '{generated_answer}'")

    # Evaluate the test answer against the generated answer with BLEU score
    bleu_score = evaluate_with_bleu(generated_answer, test_answer)

    # Evaluate the test answer against the generated answer with semantic similarity
    semantic_similarity = evaluate_with_semantic_similarity(
        generated_answer, test_answer, model, tokenizer, device)

    # Compute the weighted score
    weighted_score = compute_weighted_score(bleu_score, semantic_similarity)

    # Uncomment to display results directly
    # print(f"\nGenerated Answer (Reference): {generated_answer}")
    # print(f"Test Answer: {test_answer}")
    # print(f"BLEU Score: {bleu_score}")
    # print(f"Semantic Similarity: {semantic_similarity}")
    # print(f"Weighted Score (out of 100): {weighted_score}")


if __name__ == "__main__":
    main()
