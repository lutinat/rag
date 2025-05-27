import torch
from utils import load_model, free_model_memory
from gpu_profiler import profile_function


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"GPU Memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2:.2f} MB")


def build_prompt_from_chunks(question: str, chunks: list[str], enable_profiling: bool = False) -> list[dict]:
    """
    Build chat messages from chunks, embedding metadata in a structured format
    to enhance relevance and grounding for the LLM.
    """
    # Apply conditional profiling
    @profile_function("build_prompt_from_chunks", enabled=enable_profiling)
    def _build_prompt():
        # Build rich context blocks with metadata
        context_blocks = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            meta_lines = "\n".join(f"  - {key.capitalize()}: {value}" for key, value in meta.items() if value)
            formatted_block = (
                f"### Context {i+1}\n"
                f"**Metadata:**\n{meta_lines if meta else '  - None'}\n"
                f"**Content:**\n{chunk['text']}"
            )
            context_blocks.append(formatted_block)

        system_prompt = {
            "role": "system",
            "content": (
                "You are an expert AI assistant for Satlantis, a company in the space sector.\n "
                "Your goal is to provide accurate, concise, and context-grounded answers strictly based on the information provided.\n "
                "Answer concisely unless more detail is needed to answer accurately.\n "
                "Do not hallucinate or make up information.\n "
                "If the information is incomplete or unclear, explain how it impacts your answer. \n "
                "In cases where an answer cannot be fully derived, explain why the full answer isn't available and what additional details would be needed.\n"
                "If the question is vague or lacks necessary context, make sure to explicitly mention the uncertainty "
                "and request additional information or clarification from the user."
            )
        }

        user_prompt = {
            "role": "user",
            "content": (
                "You are provided with independent context snippets from internal documents.\n"
                "Each one includes metadata and content. Use **only** the given information to answer.\n"
                "If unsure, explain what's missing.\n\n"
                f"{'---'.join(context_blocks)}\n\n"
                f"### Question:\n{question}"
            )
        }

        return [system_prompt, user_prompt]
    
    return _build_prompt()


def generate_answer(prompt: str, pipeline_obj=None, model_name: str = None, enable_profiling: bool = False, quantization: str = None) -> str:
    """
    Generate an answer to the question based on the provided chunks.
    
    Args:
        prompt (str): The prompt to generate an answer for
        pipeline_obj: Pre-loaded pipeline object (preferred)
        model_name (str): Model name if pipeline_obj is not provided (for backward compatibility)
        enable_profiling (bool): Whether to enable profiling
        quantization (str): Quantization type if model_name is used
    """
    # Apply conditional profiling
    @profile_function("generate_answer", enabled=enable_profiling)
    def _generate_answer():
        # Use provided pipeline or load model
        if pipeline_obj is not None:
            pipeline = pipeline_obj
            should_cleanup = False
        else:
            # Backward compatibility: load model if no pipeline provided
            _, _, pipeline = load_model(model_name, quantization=quantization)
            should_cleanup = True
        
        try:
            # Define generation arguments
            generation_args = { 
                "max_new_tokens": 500, 
                "return_full_text": False, 
                "temperature": 0.1,
                "do_sample": True, 
            }
            
            # Generate the answer
            # Use try-except to handle potential errors during generation
            try:
                output = pipeline(prompt, **generation_args)
                answer = output[0]['generated_text'].strip()
            except Exception as e:
                print(f"Error during generation: {e}")
                answer = "An error occurred while generating the answer. Please try again."
            
            return answer
        
        finally:
            # Only clean up if we loaded the model ourselves
            if should_cleanup:
                free_model_memory(pipeline)
    
    return _generate_answer()