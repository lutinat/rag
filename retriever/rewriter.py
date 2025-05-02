from utils import load_model, free_model_memory


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
        "max_new_tokens": 100,  # Limite réduite pour forcer une réponse concise
        "return_full_text": False,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 0.9,  # Contrôle supplémentaire sur l'échantillonnage
    }

    prompt = (
        "You are a helpful technical assistant for a company with expertise in satellite, space sector, their teams and internal tools.\n "
        "Generate a single, concise and plausible full-sentence that could answer the following question.\n "
        "Reintroduce the question and important words and vocabulary from the question in your answer to stay on topic.\n "
        "The sentence should sound natural and informative, useful for retrieving relevant documents.\n "
        "Avoid over-speculation or unrelated details.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


    try:
        output = pipeline(prompt, **generation_args)
        answer = output[0]['generated_text'].strip()
    except Exception as e:
        raise RuntimeError(f"HyDE generation failed: {e}")
    finally:
        # Clean up the pipeline to free memory
        free_model_memory(pipeline)

    return answer