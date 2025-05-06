from utils import load_model, free_model_memory
import json
from tqdm import tqdm


def hyDE(question: str, model_name: str) -> str:
    """
    Generate a hypothetical answer to the question to be used for retrieval.
    The answer is concise and helps improve the semantic search step in RAG.

    Args:
        question (str): The question to generate an answer for.
        model_name (str): The name of the model to use for generation.
    Returns:
        str: The generated answer.
    """

    _, _, pipeline = load_model(model_name)

    if not question or not isinstance(question, str):
        raise ValueError("Invalid question provided.")

    generation_args = {
        "max_new_tokens": 60,  # Limite réduite pour forcer une réponse concise
        "return_full_text": False,
        "temperature": 0,
        # "do_sample": True,
        # "top_p": 0.9,  # Contrôle supplémentaire sur l'échantillonnage
    }

    # Define the system prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are an assistant generating internal documentation for Satlantis, a company in the space sector.\n"
            "Your task is to generate a **single, realistic, and plausible sentence** as if it were extracted from a confidential internal report.\n"
            "**Never** begin the sentence with phrases like 'The answer is', 'It is possible that', or any generic statement.\n"
            "**Do not explain** or provide context — only write the kind of sentence that could appear verbatim in a report.\n"
            "If the information is not publicly available, **make up a plausible answer** that sounds credible and grounded.\n"
            "**Never fabricate exaggerated names or achievements** — keep it neutral and fact-like.\n"
            "Use important words and vocabulary from the question in your answer to stay on topic.\n "
        )
    }

    # Construct the user prompt
    user_prompt = {
        "role": "user",
        "content": (
            f"Question: {question}\n"
            "Answer:"
        )
    }
    
    # Combine system and user prompts
    prompt = [system_prompt, user_prompt]

    # Generate the answer
    try:
        output = pipeline(prompt, **generation_args)
        answer = output[0]['generated_text'].strip()
    except Exception as e:
        raise RuntimeError(f"HyDE generation failed: {e}")
    finally:
        # Clean up the pipeline to free memory
        free_model_memory(pipeline)

    return answer
