import os
import sys
import glob

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import json
from tqdm import tqdm
from utils import load_model


def generate_qa_pairs(pipe, text, num_questions=3):
    """
    Generate multiple question and answer pairs based on the provided text using a language model.
    
    Args:   
        pipe: The language model pipeline to use for generation.
        text (str): The input text to generate a QA pair from.
        num_questions (int): The number of questions to generate.
    Returns:
        list: The generated (question, answer) pairs.
    """
    qa_pairs = []

    system_prompt_q =  {
            "role": "system",
            "content": (
                "You are an AI assistant for Satlantis, a company in the space sector. "
                "Your task is to generate a short, concise, general, and relevant question based on the given content. "
                "The question should sound natural and curious, using different interrogative forms like What, How, Why, Who, When, or Where. "
                "Do not repeat the same structure every time. "
                "Focus on facts, implications, causes, purposes, definitions, methods, places, dates, persons, or events. "
                "Never include explanations, context, or any mention of 'the text', 'this document', or similar phrases. "
                "The question should be very short, and very concise, and should not exceed 15 words. "
                "Only return a self-contained question. If nothing useful can be asked, return 'No question'."
                "The question can be basic or complex, but it should be relevant to the content."
                "You can use formal or informal tone to generate the question."
                "It should sound natural and human-like."
                "You can ask about uncommon terms or words in the context."
            )
        }
    system_prompt_a = { 
            "role": "system",
            "content": (
                "You are an AI assistant for Satlantis, a company in the space sector. "
                "Your task is to provide a clear, concise, and accurate answer to the given question based on the provided context. "
                "The answer should be factual and directly address the question. "
                "Write a complete sentence, and reintroduce the question in the answer. "
                "Keep the answer focused and relatively short (3 sentences maximum). "
                "Only include information that is directly relevant to answering the question."
            )
        }
    
    # We will add every message question to this list, so we can tell the model to avoid having the same question twice
    messages_q = [system_prompt_q]

    for i in range(num_questions):
        messages_a = [system_prompt_a] # For the answer, we will only provide current question
        is_first_pass = i == 0

        # Generate the question
        question_prompt = {
                "role": "user",
                "content": f"{text.strip()}",
        }
        
        if is_first_pass:
            messages_q.append(question_prompt)
        else:
            messages_q.append({
                "role": "user",
                "content": f"Ask another question that is as far as possible from the previous ones in terms of content and topic."
            })

        question_output = pipe(
            messages_q,
            max_new_tokens=50,
            return_full_text=False,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
        
        question = question_output[0]['generated_text'].strip()

        messages_q.append({
            "role": "assistant",
            "content": question
        })
        
        # Then generate the answer based on the question and original text
        answer_prompt = {
                "role": "user",
                "content": f"Context: {text.strip()}\n\nQuestion: {question}\n\nProvide a concise answer:"
        }
        messages_a.append(answer_prompt)

        answer_output = pipe(
            messages_a,
            max_new_tokens=150,
            return_full_text=False,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
        
        answer = answer_output[0]['generated_text'].strip()

        qa_pairs.append((question, answer))
    
    return qa_pairs


def process_jsonl_files(input_folder, output_jsonl, pipe, global_max_questions=1000, per_file_max_questions=10, per_chunk_questions=3):
    """
    Ultra memory-efficient streaming version that processes one line at a time.
    
    Args:
        input_folder: Path to folder containing JSONL files
        output_jsonl: Path to output JSONL file
        pipe: The model pipeline to use
        global_max_questions: Maximum total questions to generate
        per_file_max_questions: Maximum questions to generate per file
    """
    import random
    
    # Get all JSONL files in the input folder
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {input_folder}")
    
    # Shuffle the files for randomness
    random.shuffle(jsonl_files)
    
    total_questions = 0
    with open(output_jsonl, "w") as f_out:
        for jsonl_file in jsonl_files:
            if total_questions >= global_max_questions:
                print(f"Reached global max questions limit: {global_max_questions}")
                break
                
            print(f"Streaming {jsonl_file} (max {per_file_max_questions} questions)")
            
            # Count total lines for progress bar
            with open(jsonl_file, 'r') as f:
                total_lines = sum(1 for _ in f)
            
            file_questions = 0
            # Stream process line by line
            with open(jsonl_file, "r") as f_in:
                with tqdm(total=total_lines, desc=f"Processing {os.path.basename(jsonl_file)}") as pbar:
                    for line in f_in:
                        # Check both global and per-file limits
                        if total_questions >= global_max_questions or file_questions >= per_file_max_questions:
                            pbar.update(1)
                            continue
                            
                        try:
                            data = json.loads(line.strip())
                            text = data.get("text", "")
                            if len(text.strip()) < 20:
                                pbar.update(1)
                                continue  # skip too short texts

                            # Generate 3 QA pairs per chunk
                            qa_pairs = generate_qa_pairs(pipe, text, num_questions=per_chunk_questions)
                            
                            for question, answer in qa_pairs:
                                if (total_questions >= global_max_questions or 
                                    file_questions >= per_file_max_questions):
                                    break
                                    
                                # Write immediately to avoid memory buildup
                                f_out.write(json.dumps({
                                    "question": question, 
                                    "answer": answer, 
                                    "context": text,
                                }) + "\n")
                                f_out.flush()  # Force write to disk
                                total_questions += 1
                                file_questions += 1
                                
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line in {jsonl_file}")
                        except Exception as e:
                            print(f"Error processing line: {e}")
                        finally:
                            pbar.update(1)
            
            print(f"Generated {file_questions} questions from {os.path.basename(jsonl_file)}")
                
    print(f"Generated {total_questions} total QA pairs and saved to {output_jsonl}")


if __name__ == "__main__":
    _, _, pipe = load_model("microsoft/Phi-4-mini-instruct")

    input_folder = "/home/lucasd/code/rag/data/data1"  # Folder containing JSONL files
    output_jsonl = "/home/lucasd/code/rag/data_processing/generated_questions.jsonl"
    global_max_questions = 100000
    per_file_max_questions = 100000
    per_chunk_questions = 8

    # Use streaming with multiple QA pairs per chunk (3 questions per text chunk)
    process_jsonl_files(input_folder, output_jsonl, pipe, global_max_questions, per_file_max_questions, per_chunk_questions)
    
