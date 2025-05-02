from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import gc
import torch


def save_chunks_jsonl(chunks, path):
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

def load_chunks_jsonl(path):
    chunks = []
    with open(path, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def load_model(model_name):
    """
    Load a model from the Hugging Face model hub.
    Args:
        model_name (str): The name of the model to load.
    Returns:

        model: The loaded model.
        tokenizer: The tokenizer for the model.
        pipe: The pipeline for the model.
    """

        # Load the Phi-4-mini-instruct model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    )

    return model, tokenizer, pipe

def free_model_memory(pipeline):
    """
    Free up GPU memory.
    """
    # Clean up the pipeline to free memory
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()