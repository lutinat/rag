from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
from queue import Queue
from threading import Thread, Lock
import time
import gc
import torch
from typing import Optional, Dict, Any, Tuple
import os
from dataclasses import dataclass
from enum import Enum

# Import your existing modules
from utils import load_model, free_model_memory
from retriever.retriever import build_faiss_index
from utils import load_chunks_jsonl
from huggingface_hub import login
from dotenv import load_dotenv
from rag import rag
from config import ProductionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

app = Flask(__name__)
CORS(app)

class ModelType(Enum):
    LLM = "llm"
    EMBEDDER = "embedder" 
    RERANKER = "reranker"

@dataclass
class ModelInfo:
    model_type: ModelType
    model_name: str
    model_obj: Any = None
    tokenizer: Any = None
    pipeline: Any = None
    loaded: bool = False
    last_used: float = 0.0
    memory_usage_gb: float = 0.0

class ModelManager:
    """Manages model loading, unloading, and memory optimization."""
    
    def __init__(self, max_memory_gb: float = None):
        if max_memory_gb is None:
            max_memory_gb = ProductionConfig.MAX_GPU_MEMORY_GB
            
        self.models: Dict[str, ModelInfo] = {}
        self.model_lock = Lock()
        self.max_memory_gb = max_memory_gb
        self.current_memory_gb = 0.0
        self.preloaded_models = set()
        
        # Model configurations
        self.model_configs = {
            "llm": ModelInfo(
                model_type=ModelType.LLM,
                model_name=ProductionConfig.MODELS["llm"]
            ),
            "embedder": ModelInfo(
                model_type=ModelType.EMBEDDER,
                model_name=ProductionConfig.MODELS["embedder"]
            ),
            "reranker": ModelInfo(
                model_type=ModelType.RERANKER,
                model_name=ProductionConfig.MODELS["reranker"]
            )
        }
        
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def free_gpu_memory(self):
        """Free up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def can_load_model(self, estimated_size_gb: float) -> bool:
        """Check if we can load a model of given size."""
        return (self.current_memory_gb + estimated_size_gb) <= self.max_memory_gb
    
    def unload_least_recently_used(self, required_memory_gb: float):
        """Unload least recently used models to free memory."""
        with self.model_lock:
            loaded_models = [(k, v) for k, v in self.models.items() if v.loaded]
            loaded_models.sort(key=lambda x: x[1].last_used)
            
            for model_key, model_info in loaded_models:
                if self.can_load_model(required_memory_gb):
                    break
                self._unload_model(model_key)
    
    def _unload_model(self, model_key: str):
        """Internal method to unload a specific model."""
        model_info = self.models[model_key]
        if not model_info.loaded:
            return
            
        if model_info.pipeline:
            free_model_memory(model_info.pipeline)
            model_info.pipeline = None
        
        if model_info.model_obj:
            if hasattr(model_info.model_obj, 'cpu'):
                model_info.model_obj.cpu()
            del model_info.model_obj
            model_info.model_obj = None
            
        if model_info.tokenizer:
            del model_info.tokenizer
            model_info.tokenizer = None
        
        model_info.loaded = False
        self.current_memory_gb -= model_info.memory_usage_gb
        model_info.memory_usage_gb = 0.0
        self.free_gpu_memory()
    
    def load_model(self, model_key: str, quantization: str = None) -> ModelInfo:
        """Load a model with memory management."""
        if quantization is None:
            from config import ProductionConfig
            quantization = ProductionConfig.DEFAULT_QUANTIZATION
        
        with self.model_lock:
            if model_key not in self.models:
                self.models[model_key] = self.model_configs[model_key]
            
            model_info = self.models[model_key]
            
            if model_info.loaded:
                model_info.last_used = time.time()
                return model_info
            
            # Memory estimates
            memory_estimates = {
                "llm": ProductionConfig.MODEL_MEMORY_ESTIMATES["llm_4bit"] if quantization == "4bit" else ProductionConfig.MODEL_MEMORY_ESTIMATES["llm_8bit"],
                "embedder": ProductionConfig.MODEL_MEMORY_ESTIMATES["embedder"],
                "reranker": ProductionConfig.MODEL_MEMORY_ESTIMATES["reranker"]
            }
            
            required_memory = memory_estimates.get(model_key, 4.0)
            
            if not self.can_load_model(required_memory):
                self.unload_least_recently_used(required_memory)
            
            memory_before = self.get_gpu_memory_usage()
            
            try:
                if model_info.model_type == ModelType.LLM:
                    model, tokenizer, pipeline = load_model(
                        model_info.model_name, 
                        quantization=quantization
                    )
                    model_info.model_obj = model
                    model_info.tokenizer = tokenizer
                    model_info.pipeline = pipeline
                    
                elif model_info.model_type == ModelType.EMBEDDER:
                    from sentence_transformers import SentenceTransformer
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    embedder = SentenceTransformer(model_info.model_name, device=device)
                    model_info.model_obj = embedder
                    
                elif model_info.model_type == ModelType.RERANKER:
                    from FlagEmbedding import FlagReranker
                    
                    if torch.cuda.is_available():
                        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                        reranker_model = FlagReranker(model_info.model_name, use_fp16=True)
                        
                        if hasattr(reranker_model, 'model'):
                            reranker_model.model = reranker_model.model.cuda()
                    else:
                        reranker_model = FlagReranker(model_info.model_name, use_fp16=False)
                    
                    model_info.model_obj = reranker_model
                
                memory_after = self.get_gpu_memory_usage()
                model_info.memory_usage_gb = memory_after - memory_before
                model_info.loaded = True
                model_info.last_used = time.time()
                self.current_memory_gb += model_info.memory_usage_gb
                
                return model_info
                
            except Exception as e:
                logger.error(f"Failed to load {model_key}: {e}")
                model_info.loaded = False
                raise
    
    def get_model(self, model_key: str, quantization: str = None) -> Optional[ModelInfo]:
        """Get a model, loading it if necessary."""
        try:
            return self.load_model(model_key, quantization)
        except Exception as e:
            logger.error(f"Failed to get model {model_key}: {e}")
            return None
    
    def mark_as_preloaded(self, model_key: str):
        """Mark a model as preloaded."""
        self.preloaded_models.add(model_key)
    
    def unload_on_demand_models(self):
        """Unload models that were loaded on-demand (not preloaded)."""
        with self.model_lock:
            for model_key in list(self.models.keys()):
                if model_key not in self.preloaded_models:
                    model_info = self.models[model_key]
                    if model_info.loaded:
                        self._unload_model(model_key)
    
    def handle_memory_error_cleanup(self):
        """Handle CUDA memory errors by unloading on-demand models."""
        logger.warning("CUDA memory error detected, unloading on-demand models...")
        
        self.unload_on_demand_models()
        self.free_gpu_memory()
        
        if self.current_memory_gb > self.max_memory_gb * 0.8:
            with self.model_lock:
                loaded_models = [(k, v) for k, v in self.models.items() if v.loaded]
                loaded_models.sort(key=lambda x: x[1].last_used)
                
                for model_key, model_info in loaded_models[:1]:
                    self._unload_model(model_key)
                    if model_key in self.preloaded_models:
                        self.preloaded_models.remove(model_key)

class RAGProcessor:
    """Handles RAG processing using the updated rag.py function."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.chunk_path = ProductionConfig.CHUNK_PATH
        self.embeddings_folder = ProductionConfig.EMBEDDINGS_FOLDER
        self.index = None
        self.embedder_model = None
        self.chunks = None
        
    def preload_embeddings(self):
        """Preload embeddings and FAISS index."""
        try:
            self.chunks = load_chunks_jsonl(self.chunk_path)
            
            embedder_info = self.model_manager.get_model("embedder")
            if not embedder_info or not embedder_info.loaded:
                raise Exception("Failed to load embedder model for preloading")
            
            self.embedder_model = embedder_info.model_obj
            
            self.index, embeddings, self.chunks, _ = build_faiss_index(
                self.chunks,
                embedder_info.model_name,
                self.embeddings_folder,
                save_embeddings=False,
                enable_profiling=False,
                pre_loaded_embedder=self.embedder_model
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to preload embeddings: {e}")
            self.index = None
            self.embedder_model = None  
            self.chunks = None
            return False
            
    def process_question(self, question: str, quantization: str = None) -> Tuple[Dict[str, Any], int]:
        """Process a question using the rag.py function with preloaded models."""
        if quantization is None:
            from config import ProductionConfig
            quantization = ProductionConfig.DEFAULT_QUANTIZATION
            
        try:
            preloaded_models = {}
            preloaded_embeddings = {}
            
            # Get models
            llm_info = self.model_manager.get_model("llm", quantization)
            if llm_info and llm_info.loaded:
                preloaded_models['llm'] = llm_info.pipeline
            
            embedder_info = self.model_manager.get_model("embedder")
            if embedder_info and embedder_info.loaded:
                preloaded_models['embedder'] = embedder_info.model_obj
            
            reranker_info = self.model_manager.get_model("reranker")
            if reranker_info and reranker_info.loaded:
                preloaded_models['reranker'] = reranker_info.model_obj
            
            # Use preloaded embeddings if available
            if self.index is not None and self.embedder_model is not None:
                preloaded_embeddings = {
                    'index': self.index,
                    'embedder': self.embedder_model,
                    'chunks': self.chunks
                }
            
            # Call the rag function
            answer, sources = rag(
                question=question,
                recompute_embeddings=False,
                enable_profiling=False,
                quantization=quantization,
                preloaded_models=preloaded_models if preloaded_models else None,
                preloaded_embeddings=preloaded_embeddings if preloaded_embeddings else None
            )
            
            return {
                "answer": answer,
                "sources": sources
            }, 200
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory error: {e}")
            self.model_manager.handle_memory_error_cleanup()
            
            return {
                "error": "GPU memory is full. On-demand models have been unloaded - please try again.",
                "error_type": "memory_error"
            }, 503
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            
            error_str = str(e).lower()
            if "memory" in error_str or "cuda" in error_str:
                self.model_manager.handle_memory_error_cleanup()
                return {
                    "error": "Memory-related error occurred. Please try again.",
                    "error_type": "memory_error"
                }, 503
            
            return {
                "error": f"An error occurred while processing your question: {str(e)}",
                "error_type": "processing_error"
            }, 500

# Global variables
model_manager = None
rag_processor = None
request_queue = Queue()
worker_lock = Lock()

def worker():
    """Worker function to process requests from the queue."""
    while True:
        response_queue = None
        try:
            request_data = request_queue.get()
            if request_data is None:
                break
                
            question, quantization, response_queue = request_data
            result, status_code = rag_processor.process_question(question, quantization)
            response_queue.put((result, status_code))
            
        except Exception as e:
            logger.error(f"Error in worker: {str(e)}")
            if response_queue is not None:
                try:
                    error_response = {'error': f'Internal server error: {str(e)}'}
                    response_queue.put((error_response, 500))
                except Exception:
                    pass
        finally:
            request_queue.task_done()

@app.route('/api/question', methods=['POST'])
def question():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        quantization = data.get('quantization', None)
            
        response_queue = Queue()
        request_queue.put((question, quantization, response_queue))
        
        try:
            result, status_code = response_queue.get(timeout=ProductionConfig.REQUEST_TIMEOUT)
            return jsonify(result), status_code
        except Exception as e:
            return jsonify({'error': f'Request processing failed: {str(e)}'}), 504
            
    except Exception as e:
        logger.error(f"Unexpected error in question endpoint: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        memory_usage = model_manager.get_gpu_memory_usage() if model_manager else 0.0
        return jsonify({
            'status': 'healthy',
            'gpu_memory_gb': round(memory_usage, 2),
            'loaded_models': list(model_manager.models.keys()) if model_manager else [],
            'preloaded_models': list(model_manager.preloaded_models) if model_manager else []
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/models/status', methods=['GET'])
def model_status():
    """Get status of all models."""
    try:
        models_info = {}
        embeddings_info = {}
        
        if model_manager:
            for key, info in model_manager.models.items():
                models_info[key] = {
                    'loaded': info.loaded,
                    'memory_gb': round(info.memory_usage_gb, 2),
                    'last_used': info.last_used,
                    'preloaded': key in model_manager.preloaded_models,
                    'model_type': info.model_type.value
                }
        
        if rag_processor:
            embeddings_info = {
                'embeddings_preloaded': rag_processor.index is not None,
                'chunks_loaded': rag_processor.chunks is not None,
                'num_chunks': len(rag_processor.chunks) if rag_processor.chunks else 0,
                'embedder_preloaded': rag_processor.embedder_model is not None
            }
            
        return jsonify({
            'models': models_info,
            'embeddings': embeddings_info,
            'production_mode_ready': bool(models_info and any(info['preloaded'] for info in models_info.values()))
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emergency/cleanup', methods=['POST'])
def emergency_cleanup():
    """Emergency cleanup endpoint to free memory."""
    try:
        if model_manager:
            model_manager.handle_memory_error_cleanup()
            return jsonify({
                'message': 'Emergency cleanup completed',
                'gpu_memory_gb': round(model_manager.get_gpu_memory_usage(), 2)
            }), 200
        return jsonify({'error': 'Model manager not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def cleanup_models():
    """Cleanup preloaded models before shutdown/reload."""
    global model_manager, rag_processor
    
    try:
        if rag_processor:
            rag_processor.index = None
            rag_processor.embedder_model = None
            rag_processor.chunks = None
        
        if model_manager:
            for model_key in list(model_manager.models.keys()):
                if model_manager.models[model_key].loaded:
                    model_manager._unload_model(model_key)
            
            model_manager.preloaded_models.clear()
            model_manager.current_memory_gb = 0.0
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")

def shutdown_workers():
    """Shutdown worker thread gracefully."""
    request_queue.put(None)

def initialize_app():
    """Initialize the application with model preloading."""
    global model_manager, rag_processor
    
    try:
        from config import ProductionConfig
        
        model_manager = ModelManager(max_memory_gb=ProductionConfig.MAX_GPU_MEMORY_GB)
        
        # Preload models
        for model_key in ProductionConfig.PRELOAD_MODELS:
            model_info = model_manager.get_model(model_key)
            if model_info and model_info.loaded:
                model_manager.mark_as_preloaded(model_key)
        
        rag_processor = RAGProcessor(model_manager)
        
        # Preload embeddings if embedder is preloaded
        if "embedder" in ProductionConfig.PRELOAD_MODELS:
            rag_processor.preload_embeddings()
        
        # Start worker thread
        worker_thread = Thread(target=worker, daemon=True)
        worker_thread.start()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return False

if __name__ == '__main__':
    import atexit
    import sys
    import signal
    
    use_reloader = '--reload' in sys.argv
    
    def signal_handler(signum, frame):
        cleanup_models()
        shutdown_workers()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    atexit.register(cleanup_models)
    atexit.register(shutdown_workers)
    
    if initialize_app():
        try:
            app.run(host=ProductionConfig.API_HOST, port=ProductionConfig.API_PORT, debug=True, use_reloader=use_reloader)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            if not use_reloader:
                cleanup_models()
                shutdown_workers()
    else:
        logger.error("Failed to start server due to initialization errors")
        exit(1)