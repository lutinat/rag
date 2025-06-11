from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
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

def load_model(model_name, quantization=None):
    """
    Load a model from the Hugging Face model hub.
    Args:
        model_name (str): The name of the model to load.
        quantization (str, optional): Type of quantization to apply. 
                                     Options: "4bit", "8bit", None (default: None)
    Returns:
        model: The loaded model.
        tokenizer: The tokenizer for the model.
        pipe: The pipeline for the model.
    """
    
    # Track memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
        print(f"GPU memory before model loading: {memory_before:.2f} GB")
    
    # Configure quantization if specified
    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("🔧 Loading model with 4bit quantization...")
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        print("🔧 Loading model with 8bit quantization...")
    else:
        print("🔧 Loading model without quantization...")
    
    # Load the model with optional quantization
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Track memory after loading
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_used = memory_after - memory_before
        print(f"GPU memory after model loading: {memory_after:.2f} GB")
        print(f"📊 Model memory usage: {memory_used:.2f} GB")
        
        if quantization:
            print(f"✅ Successfully loaded {quantization} quantized model")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
    )

    return model, tokenizer, pipe

def free_model_memory(pipeline):
    """
    Free up GPU memory by cleaning up the pipeline.
    """
    try:
        # More gentle cleanup approach
        if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'cpu'):
            # Move model to CPU first to free GPU memory
            pipeline.model.cpu()
        
        # Force garbage collection before deleting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Delete the pipeline object
        del pipeline
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Warning: Error during memory cleanup: {e}")
        # Minimal cleanup fallback
        try:
            del pipeline
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()