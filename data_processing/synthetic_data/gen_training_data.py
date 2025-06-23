import os
import sys
import glob

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import json
from tqdm import tqdm
from utils import load_model

def generate_qa_pair(pipe, text):
    """
    Generate a question and answer pair based on the provided text using a language model.
    
    Args:   
        pipe: The language model pipeline to use for generation.
        text (str): The input text to generate a QA pair from.
    
    Returns:
        tuple: The generated (question, answer) pair.
    """

    # First generate the question
    question_messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for Satlantis, a company in the space sector. "
                "Your task is to generate a short, concise, general, and relevant question based on the given content. "
                "The question should sound natural and curious, using different interrogative forms like What, How, Why, Who, When, or Where. "
                "Do not repeat the same structure every time. "
                "Focus on facts, implications, causes, purposes, methods, places, persons, or events and reflect realistic user search behavior. "
                "Never include explanations, context, or any mention of 'the text', 'this document', or similar phrases. "
                "The question should be very short, and very concise, and should not exceed 15 words. "
                "Only return a self-contained question. If nothing useful can be asked, return 'No question'."
            ),
        },
        {
            "role": "user",
            "content": f"{text.strip()}",
        },
    ]

    question_output = pipe(
        question_messages,
        max_new_tokens=50,
        return_full_text=False,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    
    question = question_output[0]['generated_text'].strip()
    
    # Then generate the answer based on the question and original text
    answer_messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for Satlantis, a company in the space sector. "
                "Your task is to provide a clear, concise, and accurate answer to the given question based on the provided context. "
                "The answer should be factual and directly address the question. "
                "Write a complete sentence, and reintroduce the question in the answer. "
                "Keep the answer focused and relatively short (3 sentences maximum). "
                "Only include information that is directly relevant to answering the question."
            ),
        },
        {
            "role": "user",
            "content": f"Context: {text.strip()}\n\nQuestion: {question}\n\nProvide a concise answer:",
        },
    ]

    answer_output = pipe(
        answer_messages,
        max_new_tokens=150,
        return_full_text=False,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    
    answer = answer_output[0]['generated_text'].strip()
    
    return question, answer


def process_jsonl_files(input_folder, output_jsonl, pipe, max_questions=1000):
    """
    Ultra memory-efficient streaming version that processes one line at a time.
    
    Args:
        input_folder: Path to folder containing JSONL files
        output_jsonl: Path to output JSONL file
        pipe: The model pipeline to use
        max_questions: Maximum number of questions to generate
    """
    import random
    
    # Get all JSONL files in the input folder
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {input_folder}")
    
    # Shuffle the files for randomness
    random.shuffle(jsonl_files)
    
    s = 0
    with open(output_jsonl, "w") as f_out:
        for jsonl_file in jsonl_files:
            print(f"Streaming {jsonl_file}")
            
            # Count total lines for progress bar
            with open(jsonl_file, 'r') as f:
                total_lines = sum(1 for _ in f)
            
            # Stream process line by line
            with open(jsonl_file, "r") as f_in:
                with tqdm(total=total_lines, desc=f"Streaming {os.path.basename(jsonl_file)}") as pbar:
                    for line in f_in:
                        if s >= max_questions:
                            print(f"Reached max questions limit: {max_questions}")
                            return
                            
                        try:
                            data = json.loads(line.strip())
                            text = data.get("text", "")
                            if len(text.strip()) < 20:
                                pbar.update(1)
                                continue  # skip too short texts

                            question, answer = generate_qa_pair(pipe, text)
                            if question != "No question":
                                # Write immediately to avoid memory buildup
                                f_out.write(json.dumps({
                                    "question": question, 
                                    "answer": answer, 
                                    "context": text,
                                }) + "\n")
                                f_out.flush()  # Force write to disk
                                s += 1
                                
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line in {jsonl_file}")
                        except Exception as e:
                            print(f"Error processing line: {e}")
                        finally:
                            pbar.update(1)
                
    print(f"Generated {s} QA pairs and saved to {output_jsonl}")


if __name__ == "__main__":
    _, _, pipe = load_model("microsoft/Phi-4-mini-instruct")  # Use 4bit quantization

    input_folder = "/home/lucasd/code/rag/data/data1"  # Folder containing JSONL files
    output_jsonl = "/home/lucasd/code/rag/data_processing/generated_questions.jsonl"
    max_questions = 100

    # Use ultra streaming for minimal memory usage
    process_jsonl_files(input_folder, output_jsonl, pipe, max_questions)
    
