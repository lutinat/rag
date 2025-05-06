import os
import json
from tqdm import tqdm
from utils import load_model
import hdbscan
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt

def generate_question(pipe, text):
    """
    Generate a question based on the provided text using a language model.
    Can be used for Fine-tuning, or as a validation step.
    
    Args:   
        pipe: The language model pipeline to use for generation.
        text (str): The input text to generate a question from.
    
    Returns:
        str: The generated question.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for Satlantis, a company in the space sector. "
                "Your task is to generate a short, concise, general, and relevant question based on the given content. "
                "The question should sound natural and curious, using different interrogative forms like What, How, Why, Who, When, or Where. "
                "Do not repeat the same structure every time. "
                "Focus on facts, implications, causes, purposes, methods, places, persons, or events and reflect realistic user search behavior. "
                "Never include explanations, context, or any mention of 'the text', 'this document', or similar phrases. "
                "The question should be very short, and very consice, and should not exceed 15 words.  "
                "Only return a self-contained question. If nothing useful can be asked, return 'No question'."
            ),
        },
        {
            "role": "user",
            "content": f"{text.strip()}",
        },
    ]

    output = pipe(
        messages,
        max_new_tokens=50,
        return_full_text=False,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )

    return output[0]['generated_text'].strip()


if __name__ == "__main__":
    _, _, pipe = load_model("microsoft/Phi-4-mini-instruct")

    input_jsonl = "/home/lucasd/code/rag/processed_data/www.satlantis.com_chunks.jsonl"
    output_jsonl = "/home/lucasd/code/rag/data_processing/generated_questions.jsonl"
    max_questions = 1000

    with open(input_jsonl, "r") as f:
        lines = f.readlines()

    # shuffle the lines to randomize the order
    import random
    random.shuffle(lines)

    s = 0
    with open(output_jsonl, "w") as f_out:
        for line in tqdm(lines, desc="Generating questions"):
            data = json.loads(line)
            text = data.get("text", "")
            if len(text.strip()) < 20:
                continue  # skip too short texts

            try:
                question = generate_question(pipe, text)
                f_out.write(json.dumps({"question": question}) + "\n")
            except Exception as e:
                print(f"Error: {e}")
                continue
            s += 1
            if s > max_questions:
                break
    print(f"Generated {s} questions and saved to {output_jsonl}")
    
