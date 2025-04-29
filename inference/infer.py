import os
import torch
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"GPU Memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2:.2f} MB")

def load_mistral_pipeline(model_path: str):
    """
    Load the Mistral model and create a text generation pipeline with memory optimization.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        A tuple of (tokenizer, model)
    """
    model_path = Path(model_path)
    
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error messages
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Initial GPU memory state:")
        print_gpu_memory()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load model with reduced precision and memory optimization
        model = Transformer.from_folder(
            model_path,
            device=device,
            dtype=torch.float16,  # Use float16 to reduce memory usage
        )
        
        print("After model loading:")
        print_gpu_memory()
        
        # Additional memory optimization
        model.eval()  # Set to evaluation mode
        torch.set_grad_enabled(False)  # Disable gradient computation
        
        # Move to GPU in stages
        if torch.cuda.is_available():
            print("Moving model to GPU in stages...")
            # Move embedding layer first
            if hasattr(model, "embedding"):
                model.embedding = model.embedding.to("cuda")
                torch.cuda.empty_cache()
                print("After moving embedding:")
                print_gpu_memory()
            
            # Move transformer layers one by one
            if hasattr(model, "transformer"):
                for i, layer in enumerate(model.transformer.layers):
                    model.transformer.layers[i] = layer.to("cuda")
                    torch.cuda.empty_cache()
                    print(f"After moving layer {i}:")
                    print_gpu_memory()
            
            # Move final layer norm
            if hasattr(model, "final_layer_norm"):
                model.final_layer_norm = model.final_layer_norm.to("cuda")
                torch.cuda.empty_cache()
                print("After moving final layer norm:")
                print_gpu_memory()
        
        # Load tokenizer
        tokenizer = MistralTokenizer.from_file(f"{model_path}/tokenizer.model.v3")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"Error during model loading: {str(e)}")
        # Try to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def ask_question_with_chunks(question: str, chunks: list[str], tokenizer, model) -> str:
    """
    Ask a question using a list of context chunks with the model.

    Args:
        question: The question to ask
        chunks: A list of context strings (retrieved chunks)
        tokenizer: The Mistral tokenizer
        model: The Mistral model

    Returns:
        The model's answer
    """
    # Join chunks into a single formatted context block
    formatted_context = "\n\n".join(f"Document {i+1}:\n{chunk}" for i, chunk in enumerate(chunks))

    # Format the prompt
    prompt = f"""You are a helpful AI assistant for the company Satlantis. 
    Use the most useful information from the provided documents to answer the user's question as accurately as possible.

    Context:
    {formatted_context}

    Question: {question}

    Answer:"""

    # Generate response with memory optimization
    messages = [UserMessage(content=prompt)]
    request = ChatCompletionRequest(messages=messages, max_tokens=200)

    with torch.no_grad():  # Disable gradient computation
        with torch.amp.autocast('cuda'):  # Mixed precision inference
            # Tokenizing the question
            encoded = tokenizer.encode(question, return_tensors="pt")

            # Convert input_ids to tokens
            tokens = tokenizer.convert_ids_to_tokens(encoded[0])  # Extract tokens from the batch


            print("After token encoding:")
            print_gpu_memory()

            # Ensure tokens is a list of lists
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            if not isinstance(tokens[0], list):
                tokens = [tokens]

            print("Before generation:")
            print_gpu_memory()

            out_tokens, _ = generate(
                tokens, model,
                max_tokens=200,
                temperature=0.7,
                eos_id = tokenizer.eos_token_id

            )

            print("After generation:")
            print_gpu_memory()

            result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            return result.strip(), prompt

        

# def ask_question_with_context(question, context, tokenizer, model):
#     prompt = (
#         "You are a helpful AI assistant for the company Satlantis. Use the most useful informations provided below to answer the user's question as accurately as possible.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question:\n{question}\n\n"
#         "Answer:"
#     )
    
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=300,
#             temperature=0.7,
#             do_sample=True,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
