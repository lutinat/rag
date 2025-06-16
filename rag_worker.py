import sys
import json
import rag
import gc
import torch
import atexit
import logging
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

# Configure logging to write to a file instead of stdout
logging.basicConfig(
    filename='rag_worker.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup():
    """Ensure cleanup happens even if the process is terminated."""
    logger.debug("Running cleanup")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Register cleanup function
atexit.register(cleanup)

def main():
    logger.debug("Starting worker process")
    if len(sys.argv) < 2:
        error_msg = "No question provided"
        logger.error(error_msg)
        print(json.dumps({
            "error": error_msg
        }))
        sys.exit(1)
    
    question = sys.argv[1]
    logger.debug(f"Processing question: {question}")
    
    recompute_embeddings = '-s' in sys.argv[2:]
    enable_profiling = '-p' in sys.argv[2:]
    
    # Check for quantization flag
    quantization = None
    if '-q' in sys.argv[2:]:
        q_index = sys.argv.index('-q')
        if q_index + 1 < len(sys.argv):
            quantization = sys.argv[q_index + 1]
            if quantization not in ['4bit', '8bit']:
                error_msg = "Quantization must be '4bit' or '8bit'"
                logger.error(error_msg)
                print(json.dumps({
                    "error": error_msg
                }))
                sys.exit(1)
    
    try:
        # Capture all stdout/stderr during rag processing
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            logger.debug("Calling rag.rag function")
            answer, sources = rag.rag(question, 
                                    recompute_embeddings=recompute_embeddings,
                                    enable_profiling=enable_profiling,
                                    quantization=quantization)
        
        # Log the captured output
        logger.debug(f"Captured stdout: {stdout_capture.getvalue()}")
        logger.debug(f"Captured stderr: {stderr_capture.getvalue()}")
        
        logger.debug("Successfully generated answer")
        response = {
            "answer": answer,
            "sources": sources
        }
        # Only print the JSON response
        print(json.dumps(response))
        logger.debug("Response sent")
        
    except Exception as e:
        error_msg = f"Error in rag processing: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        print(json.dumps({
            "error": error_msg
        }))
        sys.exit(1)
    finally:
        # Ensure cleanup happens
        cleanup()

if __name__ == "__main__":
    main() 