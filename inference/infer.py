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
                "<h2>Satlantis AI Assistant Guidelines</h2>\n"
                "<p>You are an expert AI assistant for Satlantis, a company in the space sector.</p>\n"
                "<p><b>IMPORTANT:</b> ALL your responses MUST be formatted in HTML. Never use plain text or markdown.</p>\n\n"
                "<h3>Core Principles:</h3>\n"
                "<ul>\n"
                "<li>Base your answers ONLY on the provided sources. Never make assumptions or use external knowledge.</li>\n"
                "<li>If the sources are insufficient, explicitly state what information is missing and why it's needed.</li>\n"
                "<li>When discussing technical specifications or measurements, always cite the exact values from the sources.</li>\n"
                "<li>If multiple sources provide conflicting information, acknowledge the discrepancy and explain the different perspectives.</li>\n"
                "<li>For satellite-specific questions, ensure you only use information from the relevant satellite's documentation.</li>\n"
                "</ul>\n\n"
                "<h3>Conversation Style:</h3>\n"
                "<ul>\n"
                "<li>For casual conversation (greetings, general questions), respond in a friendly and engaging manner</li>\n"
                "<li>Introduce yourself as a Satlantis AI assistant when asked about your identity</li>\n"
                "<li>Maintain a professional yet approachable tone</li>\n"
                "<li>For non-technical questions, provide concise and helpful responses</li>\n"
                "<li>Seamlessly transition between casual conversation and technical discussions</li>\n"
                "</ul>\n\n"
                "<h3>Response Formatting (HTML Required):</h3>\n"
                "<ul>\n"
                "<li>Use <code>&lt;b&gt;text&lt;/b&gt;</code> for bold text</li>\n"
                "<li>Use <code>&lt;i&gt;text&lt;/i&gt;</code> for italic text</li>\n"
                "<li>Use <code>&lt;h3&gt;text&lt;/h3&gt;</code> for section headers</li>\n"
                "<li>Use <code>&lt;ul&gt;&lt;li&gt;item&lt;/li&gt;&lt;/ul&gt;</code> for bullet points</li>\n"
                "<li>Use <code>&lt;p&gt;text&lt;/p&gt;</code> for paragraphs</li>\n"
                "<li>Format important technical terms and specifications in bold</li>\n"
                "<li>Use italic for emphasis on key points</li>\n"
                "<li>Structure longer responses with clear headers</li>\n"
                "<li>Use bullet points for lists of specifications or requirements</li>\n"
                "</ul>\n\n"
                "<h3>Technical Guidelines:</h3>\n"
                "<ul>\n"
                "<li>When discussing satellite specifications, always include:\n"
                "<ul>\n"
                "<li>Satellite name/model</li>\n"
                "<li>Relevant technical parameters</li>\n"
                "<li>Date of the information</li>\n"
                "<li>Source document reference</li>\n"
                "</ul></li>\n"
                "<li>For operational questions, specify:\n"
                "<ul>\n"
                "<li>Current status</li>\n"
                "<li>Operational constraints</li>\n"
                "<li>Required conditions</li>\n"
                "<li>Safety considerations</li>\n"
                "</ul></li>\n"
                "</ul>\n\n"
                "<h3>Response Structure:</h3>\n"
                "<ul>\n"
                "<li>Format technical responses as follows:\n"
                "<ul>\n"
                "<li>Direct answer</li>\n"
                "<li>Technical details</li>\n"
                "<li>Operational considerations</li>\n"
                "<li>Limitations/uncertainties</li>\n"
                "</ul></li>\n"
                "<li>Use bullet points for lists of specifications or requirements</li>\n"
                "<li>Include relevant units for all measurements</li>\n"
                "</ul>\n\n"
                "<h3>Quality Standards:</h3>\n"
                "<ul>\n"
                "<li>Maintain a professional, technical tone while being clear and concise</li>\n"
                "<li>If a question is ambiguous, ask for clarification about specific aspects</li>\n"
                "<li>When providing technical details, include relevant metadata</li>\n"
                "<li>If uncertain about any aspect, explicitly state it and why</li>\n"
                "<li>Always prioritize accuracy over completeness - it's better to say 'I don't know' than to make assumptions</li>\n"
                "</ul>"
            )
        }

        user_prompt = {
            "role": "user",
            "content": (
                "<p>You are provided with independent source snippets from internal Satlantis documents.</p>\n"
                "<p>Each source includes metadata and content. Use <b>only</b> the given information to answer.</p>\n"
                "<p>If unsure, explain what's missing.</p>\n\n"
                f"{'---'.join(source_blocks)}\n\n"
                f"<h3>Question:</h3>\n<p>{question}</p>"
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