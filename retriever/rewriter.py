import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, free_model_memory
from gpu_profiler import profile_function
import json
from tqdm import tqdm
import random
import re


def rewrite(question: str, pipeline_obj=None, model_name: str = None, enable_profiling: bool = False, quantization: str = None) -> str:
    """
    Rewrite and improve a query to make it more professional, clear, and grammatically correct.
    
    Args:
        query (str): The original query to be rewritten and improved
        pipeline_obj: Pre-loaded pipeline object (preferred)
        model_name (str): Model name if pipeline_obj is not provided (for backward compatibility)
        enable_profiling (bool): Whether to enable profiling
        quantization (str): Quantization type if model_name is used
        
    Returns:
        str: The rewritten, improved question
    """
    # Apply conditional profiling
    @profile_function("rewrite", enabled=enable_profiling)
    def _rewrite():
        # Use provided pipeline or load model
        if pipeline_obj is not None:
            pipeline = pipeline_obj
            should_cleanup = False
        else:
            # Backward compatibility: load model if no pipeline provided
            _, _, pipeline = load_model(model_name, quantization=quantization)
            should_cleanup = True
        
        try:
            if not question or not isinstance(question, str):
                raise ValueError("Invalid query provided.")

            generation_args = {
                "max_new_tokens": 80,  # Sufficient tokens for query rewriting
                "return_full_text": False,
            }

            system_prompt = {
                "role": "system",
                "content": (
                    """You are a query rewriter for Satlantis, a company in the space sector. Your job is to improve grammar and formality while preserving technical accuracy.

                    === PRIMARY GOAL ===
                    Fix obvious typos and make queries more formal and professional tone, but NEVER change technical terms or meanings.

                    === WHAT TO FIX ===
                    ✅ Common word typos: "imge" → "image", "diference" → "difference"
                    ✅ Grammar: "what s" → "what is", "can u" → "can you"
                    ✅ Formality: "c'est quoi" → "qu'est-ce que", "what's" → "what is"
                    ✅ Capitalization: proper sentence structure
                    ✅ General questions: Add context and details to make them more specific and informative

                    === WHAT TO NEVER CHANGE ===
                    ❌ Uncommon words: talisman...
                    ❌ Technical terms (keep exactly as written): psf, isim, uhr, gsd, grd, l1d, mtf, toa, geisat, talisman, garai, etc.
                    ❌ Unknown acronyms or product names
                    ❌ Word relationships and meaning
                    ❌ Technical questions (keep them precise and focused)

                    === CRITICAL RULES ===
                    1. When unsure about any word → keep it unchanged
                    2. Unknown 3-letter combinations → keep exactly as written
                    3. Don't substitute similar-looking technical terms
                    4. Preserve original language (French stays French, etc.)
                    5. Output ONLY the rewritten query - no labels, arrows, or extra text
                    6. For general questions: add relevant context about Satlantis' business areas
                    7. For technical questions: keep minimal and precise

                    === EXAMPLES ===

                    Typo Corrections:
                    how psf afect imge quality? → How does PSF affect image quality?
                    what s the diference betwen l1c and l1d? → What are the differences between L1C and L1D?
                    is rpc included in l1d? → Is RPC included in L1D?
                    can u get toa from uhr? → Is it possible to get TOA from UHR?

                    Formality (French):
                    c'est quoi le grd? → Qu'est-ce que le GRD?
                    comment ça marche? → Comment cela fonctionne-t-il?

                    Formality (English):
                    what's the diff between? → What is the difference between?
                    how come it works? → Why does it work?

                    Keep Technical Terms Unchanged:
                    what is the pmp filter? → What is the pmp filter?
                    c'est quoi le aop? → Qu'est-ce que le aop?

                    Enhance General Questions (add context):
                    what is satlantis? → What is Satlantis and what are their main products and services?
                    what satlantis do? → What does Satlantis do as a company in the satellite industry?
                    who are your customers? → Who are Satlantis' main customers and target markets?
                    how to buy products? → How can I purchase satellite imaging products or sensors from Satlantis?

                    Keep Technical Questions Minimal:
                    what is psf? → What is PSF?
                    how does mtf work? → How does MTF work?
                    is rpc included in l1d? → Is RPC included in L1D?

                    Chatbot Queries:
                    hi → Hi!
                    hello → Hello!

                    === WRONG BEHAVIORS TO AVOID ===
                    1) NEVER modify words that are not common:
                    ❌ "pmp" → "PSF" or "PPM" (keep as "pmp")
                    ❌ "aop" → "AOD" (keep as "aop") 
                    ❌ "geisat" → "geosat" (keep as "geisat")

                    2) NEVER flip the order of technical terms or product name, keep original meaning:
                    ❌ "is rpc in l1d?" → "is l1d in rpc?" (don't flip order)

                    Remember: When in doubt, change nothing. Your output should be ONLY the corrected query.
                    """  
                )
            }

            # Construct the user prompt
            user_prompt = {
                "role": "user",
                "content": f"{question}"
            }
            
            # Combine system and user prompts
            prompt = [system_prompt, user_prompt]

            # Generate the rewritten question
            try:
                output = pipeline(prompt, **generation_args)
                raw_output = output[0]['generated_text'].strip()
                
                # Extract only the rewritten part after "Rewritten:"
                if "Rewritten:" in raw_output:
                    rewritten_question = raw_output.split("Rewritten:")[-1].strip()
                else:
                    # Fallback: use the raw output if format is unexpected
                    rewritten_question = raw_output
                    
            except Exception as e:
                raise RuntimeError(f"query rewriting failed: {e}")
            
            return rewritten_question
        
        finally:
            # Only clean up if we loaded the model ourselves
            if should_cleanup:
                free_model_memory(pipeline)
    
    return _rewrite()


def hyDE(question: str, pipeline_obj=None, model_name: str = None, enable_profiling: bool = False, quantization: str = None) -> str:
    """
    Generate a hypothetical answer to the query to be used for retrieval.
    
    Args:
        query(str): The query to generate an answer for
        pipeline_obj: Pre-loaded pipeline object (preferred)
        model_name (str): Model name if pipeline_obj is not provided (for backward compatibility)
        enable_profiling (bool): Whether to enable profiling
        quantization (str): Quantization type if model_name is used
    """
    # Apply conditional profiling
    @profile_function("hyDE", enabled=enable_profiling)
    def _hyDE():
        # Use provided pipeline or load model
        if pipeline_obj is not None:
            pipeline = pipeline_obj
            should_cleanup = False
        else:
            # Backward compatibility: load model if no pipeline provided
            _, _, pipeline = load_model(model_name, quantization=quantization)
            should_cleanup = True
        
        try:
            if not question or not isinstance(question, str):
                raise ValueError("Invalid query provided.")

            generation_args = {
                "max_new_tokens": 60,  # Limite réduite pour forcer une réponse concise
                "return_full_text": False,
                #"temperature": 0.1,
                # "do_sample": True,
                # "top_p": 0.9,
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
            
            return answer
        
        finally:
            # Only clean up if we loaded the model ourselves
            if should_cleanup:
                free_model_memory(pipeline)
    
    return _hyDE()


if __name__ == "__main__":
    import json
    
    def introduce_typos(text, typo_rate=0.3):
        """Introduce realistic typos: remove letters, swap letters, space issues, letter substitutions"""
        words = text.split()
        
        # Common keyboard substitutions (nearby keys)
        substitutions = {
            'a': 's', 'b': 'v', 'c': 'x', 'd': 's', 'e': 'r', 'f': 'g', 'g': 'h', 'h': 'j',
            'i': 'o', 'j': 'k', 'k': 'l', 'l': 'k', 'm': 'n', 'n': 'm', 'o': 'p', 'p': 'o',
            'q': 'w', 'r': 't', 's': 'd', 't': 'y', 'u': 'i', 'v': 'b', 'w': 'q', 'x': 'z',
            'y': 't', 'z': 'x'
        }
        
        typos_made = 0
        target_typos = max(1, int(len(words) * typo_rate))
        
        # First, handle space-related typos (compound words, extra spaces)
        space_typos = 0
        max_space_typos = min(2, target_typos // 2)  # Limit space typos
        
        if len(words) > 1 and space_typos < max_space_typos and random.random() < 0.3:
            # Remove space between two words (create compound word)
            pos = random.randint(0, len(words) - 2)
            if len(words[pos]) > 2 and len(words[pos + 1]) > 2:
                words[pos] = words[pos] + words[pos + 1]
                words.pop(pos + 1)
                space_typos += 1
                typos_made += 1
        
        # Process individual words for other typo types
        modified_words = []
        for i, word in enumerate(words):
            # Skip very short words and technical terms
            if len(word) <= 3 or word.isupper() or any(char in word for char in ['-', '/', '.']):
                modified_words.append(word)
                continue
            
            # Apply different types of typos if we haven't reached target
            if typos_made < target_typos and len(word) > 4:
                typo_type = random.choice([
                    'remove_letter', 'swap_letters', 'double_letter', 
                    'substitute_letter', 'extra_space'
                ])
                original_word = word
                
                if typo_type == 'remove_letter':
                    # Remove random letter from middle
                    remove_pos = random.randint(1, len(word) - 2)
                    word = word[:remove_pos] + word[remove_pos + 1:]
                    
                elif typo_type == 'swap_letters' and len(word) > 5:
                    # Swap two adjacent letters
                    pos = random.randint(1, len(word) - 3)
                    word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                    
                elif typo_type == 'double_letter' and len(word) > 4:
                    # Double a letter
                    pos = random.randint(1, len(word) - 2)
                    word = word[:pos] + word[pos] + word[pos:]
                    
                elif typo_type == 'substitute_letter':
                    # Substitute a letter with nearby keyboard key
                    pos = random.randint(1, len(word) - 2)
                    old_char = word[pos].lower()
                    if old_char in substitutions:
                        new_char = substitutions[old_char]
                        # Preserve original case
                        if word[pos].isupper():
                            new_char = new_char.upper()
                        word = word[:pos] + new_char + word[pos+1:]
                        
                elif typo_type == 'extra_space' and len(word) > 6:
                    # Split word with extra space
                    split_pos = random.randint(2, len(word) - 3)
                    word = word[:split_pos] + ' ' + word[split_pos:]
                
                if word != original_word:
                    typos_made += 1
            
            # Handle extra space case (creates two items)
            if ' ' in word:
                modified_words.extend(word.split())
            else:
                modified_words.append(word)
        
        # If no typos were made, force at least one
        if typos_made == 0:
            for i, word in enumerate(modified_words):
                if len(word) > 4 and not word.isupper() and not any(char in word for char in ['-', '/', '.']):
                    # Simple letter removal as fallback
                    remove_pos = random.randint(1, len(word) - 2)
                    modified_words[i] = word[:remove_pos] + word[remove_pos + 1:]
                    break
        
        return ' '.join(modified_words)
    
    def print_comparison(original, typo, rewritten, test_num):
        """Print a simple comparison of the rewrite results"""
        print(f"\n{'='*60}")
        print(f"TEST {test_num + 1}")
        print(f"{'='*60}")
        print(f"❌ WITH TYPOS: {typo}")
        print(f"✅ CORRECTED:  {rewritten}")
        
        # Check if typos were actually introduced
        if typo == original:
            print(f"⚠️  WARNING: No typos were introduced!")
        elif typo == rewritten:
            print(f"⚠️  WARNING: No correction was made!")
    
    # Load random questions from the JSON file
    jsonl_file = "/home/lucasd/code/rag/data_processing/synthetic_data/generated_questions.jsonl"
    questions = []
    
    print("📂 Loading questions from dataset...")
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                questions.append(data['question'])
        print(f"✅ Loaded {len(questions)} questions from dataset")
    except Exception as e:
        print(f"❌ Error reading questions file: {e}")
        questions = ["What is SATLANTIS?"]  # fallback
    
    # Select random questions and introduce typos
    random.seed(42)  # For reproducible results
    num_tests = 6  # Number of query pairs to test
    selected_questions = random.sample(questions, min(num_tests, len(questions)))
    
    print(f"🎲 Selected {len(selected_questions)} random questions for testing")
    print(f"🔧 Introducing typos with {0.3*100}% error rate (minimum 1 typo per question)...")
    
    test_pairs = []
    for q in selected_questions:
        typo_q = introduce_typos(q)
        # Verify typos were actually made
        if typo_q == q:
            print(f"⚠️  Warning: No typos made for: {q[:50]}...")
            # Force a typo by modifying the first suitable word
            words = q.split()
            for i, word in enumerate(words):
                if len(word) > 4:
                    words[i] = word[:-1]  # Remove last letter
                    typo_q = ' '.join(words)
                    break
        test_pairs.append((q, typo_q))
    
    model_name = "microsoft/Phi-4-mini-instruct"
    quantization = "4bit"
    
    print(f"\n🤖 Testing query Rewriter:")
    print(f"📊 Model: {model_name} ({quantization})")
    print("=" * 70)
    
    # Load model once
    print("🔄 Loading model...")
    try:
        _, _, pipeline = load_model(model_name, quantization=quantization)
        print("✅ Model loaded successfully!")
    
        manual_questions = [
            "c'est quoi le aot?",
            "commment faire un pipescan sur de la bab dans la joinlist",
            "what is satlantis?",
            "c'est quoi le aop",
            "quels sont les specification de geisat precusor",
            "comment fonctionne le psf",
            "salut, comment fonctionne le psf",
            "hi",
            "what is the pmp filter",
            "what is the grd of talisman, geisar, garai and uhr neo?",
            "what are the key features of isim 90/170 and satellite acquisition. Volum, mass, gsd, bands, Dwell time, BAck scanning, swath, pushframe, Los modos de adquisición (que deben ser bright, contrast, precision)",
            "c'est quoi la mission talisman?",
            "how many bands does geisat have?",
            "quelle est la résolution de uhr neo ?",
            "is rpc included in l1d?",
            "how to access satlantis images?",
            "how uhr impacts gsd?",
            "c'est quoi le produit l1d ?",
            "do you provide cloud processing?",
            "what's the difference between isim 90 and isim 170?",
            "which camera is used in garai?",
            "what type of missions has satlantis launched?",
            "are images corrected for brdf?",
            "wat is satlantis main product?",
            "do u suport on demand taksing?",
            "how dose psf afect the sharpnes?",
            "c'est quoi le geisat precuser?",
            "do you buil cams or full sats?",
            "hi, can i acces demo imags?",
            "what is the diference btw l1c and l1d?",
            "how many mssions hav u launched?",
            "can i get toa reflectnce from uhr?",
            "whats the swath size of isim 90?",
            "is mtf befor or after uhr procesing?",
            "quel est le format d'image brut?",
            "is bright mode better for vegetation?",
            "can isim backscan in dark condtions?",
            "how u handle temp contrl on cams?",
            "do satlantis sell raw data?",
            "what reslution can i get from garai?",
            "how fast can i get the image after taksing?",
            "c'est quoi la diff entre isim 90 et 170?",
            "does uhr neo work with any satellite bus?",
            "how do you ensure geometric consistency between scenes?",
            "what's the difference between nadir and off-nadir acquisitions in your system?",
            "hi, can i access geisat demo data?",
            "quelle est la durée de vie des missions actuelles?",
            "do you offer stereo image acquisition for 3D reconstruction?",
            "is there a cloud API to task acquisitions?"
        ]

        i = 0
        for q in manual_questions:
            rewritten = rewrite(q, pipeline_obj=pipeline)
            print_comparison(q, q, rewritten, i)
            i += 1
        
        # Test all query pairs
        successful_corrections = 0
        total_tests = len(test_pairs)
        
        for i, (original, typo) in enumerate(test_pairs):
            try:
                print(f"\n⏳ Processing test {i+1}/{total_tests}...")
                rewritten = rewrite(typo, pipeline_obj=pipeline)
                
                print_comparison(original, typo, rewritten, i)
                
                # Count successful corrections (simple check)
                if rewritten.lower().strip() != typo.lower().strip():
                    successful_corrections += 1
                    
            except Exception as e:
                print(f"❌ Error processing query{i+1}: {e}")
        
        # Simple final summary
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY: {successful_corrections}/{total_tests} questions were modified by the rewriter")
        print(f"{'='*60}")
                
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
    finally:
        # Clean up model
        try:
            free_model_memory(pipeline)
            print("\n🧹 Model memory cleaned up.")
        except:
            pass