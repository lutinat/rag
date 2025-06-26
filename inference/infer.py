import torch
from utils import load_model, free_model_memory
from gpu_profiler import profile_function
from datetime import datetime

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"GPU Memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2:.2f} MB")


def build_prompt_from_chunks(question: str, 
                             chunks: list[str], 
                             enable_profiling: bool = False,
                             conversation_history: list[dict] = None) -> list[dict]:
    """
    Build chat messages from chunks, embedding metadata in a structured format
    to enhance relevance and grounding for the LLM.
    """
    # Apply conditional profiling
    @profile_function("build_prompt_from_chunks", enabled=enable_profiling)
    def _build_prompt():

        system_prompt = {
            "role": "system",
            "content": (
                "You are Chatlantis, the expert AI assistant chatbot for Satlantis, and developped by Satlantis France, a company in the space sector.\n"
                "Chatlantis's goal is to provide accurate, concise, and context-grounded answers strictly based on the information provided.\n"
                "Your responses must follow these strict guidelines:\n\n"
                "Chatlantis's Core Principles:\n"
                "1. Base your answers ONLY on the provided sources. Never make assumptions or use external knowledge.\n"
                "2. If the sources are insufficient, explicitly state what information is missing and why it's needed.\n"
                "3. When discussing technical specifications or measurements, always cite the exact values from the sources.\n"
                "4. If multiple sources provide conflicting information, acknowledge the discrepancy and explain the different perspectives.\n"
                "5. You must provide details why you answer the question, and why you chose the sources you did.\n"
                "6. You must discuss about your answer in a few words, and provide limitations and uncertainties.\n"
                "7. For satellite-specific questions, ensure you only use information from the relevant satellite's documentation.\n\n"
                "8. You are provided with a conversation history if it exists. Use it to understand the context of the question and answer the question accordingly.\n"
                "Chatlantis's Conversation Style:\n"
                "1. For casual conversation (greetings, general questions), respond in a friendly and engaging manner\n"
                "2. Introduce yourself as Chatlantis, an AI assistant for Satlantis developped by Satlantis France, when asked about your identity\n"
                "3. Maintain a professional yet approachable tone\n"
                "4. For non-technical questions, provide concise and helpful responses\n"
                "5. Seamlessly transition between casual conversation and technical discussions\n\n"
                "Chatlantis's Response Formatting:\n"
                "1. Use HTML tags for text formatting in your responses:\n"
                "   - <b>text</b> for bold text\n"
                "   - <i>text</i> for italic text\n"
                "   - <h3>text</h3> for section headers\n"
                "   - <ul><li>item</li></ul> for bullet points\n"
                "   - <p>text</p> for paragraphs\n"
                "2. Format important technical terms and specifications in bold\n"
                "3. Use italic for emphasis on key points\n"
                "4. Use bullet points for lists of specifications or requirements\n\n"
                "Chatlantis's Technical Guidelines:\n"
                "1. When discussing satellite specifications, always include:\n"
                "   - Satellite name/model\n"
                "   - Relevant technical parameters\n"
                "   - Date of the information\n"
                "   - Source document reference\n"
                "2. For operational questions, specify:\n"
                "   - Current status\n"
                "   - Operational constraints\n"
                "   - Required conditions\n"
                "   - Safety considerations\n\n"
                "Chatlantis's Response Structure:\n"
                "1. Format technical responses as follows:\n"
                "   - Direct answer\n"
                "   - Technical details\n"
                "   - Operational considerations\n"
                "   - Limitations/uncertainties\n"
                "2. Use bullet points for lists of specifications or requirements\n"
                "3. Include relevant units for all measurements\n\n"
                "Chatlantis's Quality Standards:\n"
                "1. Maintain a professional, technical tone while being clear and concise\n"
                "2. If a question is ambiguous, ask for clarification about specific aspects\n"
                "3. When providing technical details, include relevant metadata\n"
                "4. If uncertain about any aspect, explicitly state it and why\n"
                "5. Each source is independent, please analyze each source separately before answering the question and do not make assumptions based on other sources.\n"
                "6. Ensure the answer directly addresses the question. Return only information that is explicitly related to the requested parameter in the source.\n"
                "Example:\n"
                "If the question asks for the spectral bands of Sentinel-2, do not return bands from Landsat-8 or generic band information. Only return bands clearly linked to Sentinel-2 in the source.\n"
                "7. Always prioritize accuracy over completeness - it's better to say 'I don't know' than to make assumptions\n"
                "8. Be brief, concise and to the point. Don't be redundant. Do not use unnecessary words.\n"
                "9. You must answer using the same language as the question.\n"
                "\n"
                "CRITICAL ENTITY DISAMBIGUATION RULES:\n"
                "10. When multiple similar entities are mentioned (Satellite, missions, bands...), you MUST:\n"
                "    - Identify the exact entity mentioned in the question\n"
                "    - ONLY use information explicitly linked to that specific entity\n"
                "    - If a source mentions multiple entities, clearly separate which information belongs to which\n"
                "    - Never transfer properties/dates/specifications from one entity to another\n"
                "    - If unsure which entity a piece of information refers to, state this uncertainty\n"
                "11. For entity specific questions, verify the entity name matches exactly before using any technical data\n"
                "12. If sources contain information about multiple entities but question asks about one specific entity, ignore information about other entities completely\n"
                "\n"
                "Additional information:\n"
                "- Current date and time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
        }

        messages = [system_prompt]
        
        # Add conversation history as separate messages if it exists
        if conversation_history:
            for turn in conversation_history:
                messages.append({"role": "user", "content": turn['user']})
                messages.append({"role": "assistant", "content": turn['assistant']})
    
        # Add context as assistant message
        # Build rich context blocks with metadata
        source_blocks = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            meta_lines = "\n".join(f"  - {key.capitalize()}: {value}" for key, value in meta.items() if value)
            formatted_block = (
                f"Source {i+1}\n"
                f"Metadata:\n{meta_lines if meta else '  - None'}\n"
                f"Content:\n{chunk['text']}"
            )
            source_blocks.append(formatted_block)
        context_prompt = (
            f"All {len(source_blocks)} Sources (independent sources):\n"
            "You are provided with independent source snippets from internal Satlantis documents.\n"
            "Each source includes metadata and content. Use only the given information to answer.\n\n"
            "You can also use the conversation history to understand the context of the question and answer the question accordingly.\n\n"
            "Format explanation:\n"
            "- Each source is numbered (Source X)\n"
            "- Metadata: section contains document properties (file name, page, section, etc.)\n"
            "- Content: section contains the actual text from the document\n"
            "- Sources are separated by '---' dividers\n"
            "- All sources are independent - analyze each separately\n\n"
            f"{'---'.join(source_blocks)}\n\n"
        )
        context_message = {
            "role": "assistant",
            "content": context_prompt
        }
        messages.append(context_message)

        # Add question as user message     
        user_message = (
            "Using the sources provided, answer the following question concisely using HTML formatting, and using the same language as the question:\n"
        )
        user_prompt = {
            "role": "user",
            "content": user_message + question
        }
        
        messages.append(user_prompt)

        return messages
    
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
                "temperature": 0.05,
                "do_sample": True, 
            }
            
            # Generate the answer
            # Use try-except to handle potential errors during generation
            try:
                output = pipeline(prompt, **generation_args)
                answer = output[0]['generated_text'].strip()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of memory error: {e}")
                    answer = "Out of memory error, not enough GPU memory available. Please try again with a simpler question."
                    # Optional: clear memory
                    torch.cuda.empty_cache()
                else:
                    print(f"Runtime error: {e}")
                    answer = "A runtime error occurred. Please try again."

            except Exception as e:
                print(f"Error during generation: {e}")
                answer = "An error occurred while generating the answer. Please try again."

            return answer
        
        finally:
            # Only clean up if we loaded the model ourselves
            if should_cleanup:
                free_model_memory(pipeline)
    
    return _generate_answer()