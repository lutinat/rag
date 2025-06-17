from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import subprocess
import json
import gc
import torch
import os
import signal
import atexit
import logging
import traceback
from queue import Queue
from threading import Thread, Lock
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global queue and worker pool
request_queue = Queue()
MAX_WORKERS = 1  # Changed to 1 for sequential processing
active_workers = 0
worker_lock = Lock()

def cleanup():
    """Ensure cleanup happens when the API process exits."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Register cleanup function
atexit.register(cleanup)

def process_request(question):
    """Process a single request using the worker script."""
    try:
        # Create a new process group for the subprocess
        process = subprocess.Popen(
            ['python', 'rag_worker.py', question, '-q', '4bit'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Wait for the process to complete with a timeout
        stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
        
        if process.returncode != 0:
            logger.error(f"Worker process failed with return code {process.returncode}")
            logger.error(f"Worker stderr: {stderr}")
            return {'error': f'Worker process failed: {stderr}'}, 500
        
        # Parse the JSON output from the worker
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse worker output as JSON: {e}")
            logger.error(f"Raw output: {stdout}")
            return {'error': 'Invalid response from worker process'}, 500
        
        if 'error' in response:
            logger.error(f"Worker reported error: {response['error']}")
            return {'error': response['error']}, 500
            
        return {
            'answer': response['answer'],
            'sources': response['sources']
        }, 200
        
    except subprocess.TimeoutExpired:
        logger.error("Worker process timed out")
        # Kill the entire process group if timeout occurs
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return {'error': 'Request timed out after 5 minutes'}, 504
    except Exception as e:
        logger.error(f"Error in subprocess handling: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': f'Error processing request: {str(e)}'}, 500

def worker():
    """Worker function to process requests from the queue."""
    global active_workers
    
    while True:
        response_queue = None
        try:
            # Get a request from the queue
            request_data = request_queue.get()
            if request_data is None:  # Poison pill to stop the worker
                break
                
            question, response_queue = request_data
            logger.info(f"Processing request: {question}")
            
            # Process the request
            result, status_code = process_request(question)
            
            # Put the result in the response queue
            response_queue.put((result, status_code))
            logger.info("Request processing completed")
            
        except Exception as e:
            logger.error(f"Error in worker: {str(e)}")
            logger.error(traceback.format_exc())
            # Ensure we always put something in the response queue
            if response_queue is not None:
                try:
                    error_response = {'error': f'Internal server error: {str(e)}'}
                    response_queue.put((error_response, 500))
                except Exception as queue_error:
                    logger.error(f"Failed to put error response in queue: {queue_error}")
        finally:
            request_queue.task_done()
            with worker_lock:
                active_workers -= 1

# Start worker thread
worker_thread = Thread(target=worker, daemon=True)
worker_thread.start()

@app.route('/api/question', methods=['POST'])
def question():
    logger.debug("Received question request")
    try:
        data = request.json
        if not data:
            logger.error("No JSON data in request")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        question = data.get('question')
        if not question:
            logger.error("No question in request data")
            return jsonify({'error': 'No question provided'}), 400
            
        logger.debug(f"Queueing question: {question}")
        
        # Create a response queue for this request
        response_queue = Queue()
        
        # Add request to the queue
        request_queue.put((question, response_queue))
        
        # Wait for the response with better error handling
        try:
            result, status_code = response_queue.get(timeout=300)
            logger.debug(f"Received response with status code: {status_code}")
            return jsonify(result), status_code
        except Exception as e:
            error_msg = str(e) if str(e) else "Empty response from worker"
            logger.error(f"Error waiting for response: {error_msg}")
            return jsonify({'error': f'Request processing failed: {error_msg}'}), 504
            
    except Exception as e:
        logger.error(f"Unexpected error in question endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Unexpected error: {str(e)}'
        }), 500
    finally:
        # Clean up any remaining resources
        cleanup()

def shutdown_workers():
    """Shutdown worker thread gracefully."""
    # Add poison pill to stop worker
    request_queue.put(None)
    # Wait for worker to finish
    worker_thread.join()

# Register shutdown function
atexit.register(shutdown_workers)

if __name__ == '__main__':
    logger.info("Starting API server with sequential processing...")
    app.run(host='0.0.0.0', port=5000, debug=True)
