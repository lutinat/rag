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
        source_blocks = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            meta_lines = "\n".join(f"  - {key.capitalize()}: {value}" for key, value in meta.items() if value)
            formatted_block = (
                f"### Source {i+1}\n"
                f"**Metadata:**\n{meta_lines if meta else '  - None'}\n"
                f"**Content:**\n{chunk['text']}"
            )
            source_blocks.append(formatted_block)

        system_prompt = {
            "role": "system",
            "content": (
                "You are an expert AI assistant for Satlantis, a company in the space sector.\n"
                "Your responses must follow these strict guidelines:\n\n"
                "Core Principles:\n"
                "1. Base your answers ONLY on the provided sources. Never make assumptions or use external knowledge.\n"
                "2. If the sources are insufficient, explicitly state what information is missing and why it's needed.\n"
                "3. When discussing technical specifications or measurements, always cite the exact values from the sources.\n"
                "4. If multiple sources provide conflicting information, acknowledge the discrepancy and explain the different perspectives.\n"
                "5. For satellite-specific questions, ensure you only use information from the relevant satellite's documentation.\n\n"
                "Conversation Style:\n"
                "1. For casual conversation (greetings, general questions), respond in a friendly and engaging manner\n"
                "2. Introduce yourself as a Satlantis AI assistant when asked about your identity\n"
                "3. Maintain a professional yet approachable tone\n"
                "4. For non-technical questions, provide concise and helpful responses\n"
                "5. Seamlessly transition between casual conversation and technical discussions\n\n"
                "Response Formatting:\n"
                "1. Use HTML tags for text formatting in your responses:\n"
                "   - <b>text</b> for bold text\n"
                "   - <i>text</i> for italic text\n"
                "   - <h3>text</h3> for section headers\n"
                "   - <ul><li>item</li></ul> for bullet points\n"
                "   - <p>text</p> for paragraphs\n"
                "2. Format important technical terms and specifications in bold\n"
                "3. Use italic for emphasis on key points\n"
                "4. Structure longer responses with clear headers\n"
                "5. Use bullet points for lists of specifications or requirements\n\n"
                "Technical Guidelines:\n"
                "6. When discussing satellite specifications, always include:\n"
                "   - Satellite name/model\n"
                "   - Relevant technical parameters\n"
                "   - Date of the information\n"
                "   - Source document reference\n"
                "7. For operational questions, specify:\n"
                "   - Current status\n"
                "   - Operational constraints\n"
                "   - Required conditions\n"
                "   - Safety considerations\n\n"
                "Response Structure:\n"
                "8. Format technical responses as follows:\n"
                "   - Direct answer\n"
                "   - Technical details\n"
                "   - Operational considerations\n"
                "   - Limitations/uncertainties\n"
                "9. Use bullet points for lists of specifications or requirements\n"
                "10. Include relevant units for all measurements\n\n"
                "Quality Standards:\n"
                "11. Maintain a professional, technical tone while being clear and concise\n"
                "12. If a question is ambiguous, ask for clarification about specific aspects\n"
                "13. When providing technical details, include relevant metadata\n"
                "14. If uncertain about any aspect, explicitly state it and why\n"
                "15. Always prioritize accuracy over completeness - it's better to say 'I don't know' than to make assumptions"
            )
        }

        user_prompt = {
            "role": "user",
            "content": (
                "You are provided with independent source snippets from internal Satlantis documents.\n"
                "Each source includes metadata and content. Use **only** the given information to answer.\n"
                "If unsure, explain what's missing.\n\n"
                f"{'---'.join(source_blocks)}\n\n"
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