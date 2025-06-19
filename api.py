from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
from queue import Queue
from threading import Thread, Lock
import threading
import time
import gc
import torch
from typing import Optional, Dict, Any, Tuple, List
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
            
    def process_question(self, question: str, quantization: str = None, conversation_history: List[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
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
                preloaded_embeddings=preloaded_embeddings if preloaded_embeddings else None,
                conversation_history=conversation_history
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

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    user_message: str
    assistant_response: str
    timestamp: float
    sources: Optional[List[str]] = None

@dataclass
class ChatSession:
    """Represents a chat session with metadata."""
    chat_id: str
    created_at: float
    last_activity: float
    turns: List[ConversationTurn]

class ConversationManager:
    """Manages conversation history per chat session."""
    
    def __init__(self, max_history_per_chat: int = 20, max_inactive_hours: int = 24):
        self.chat_sessions: Dict[str, ChatSession] = {}
        self.max_history = max_history_per_chat
        self.max_inactive_hours = max_inactive_hours
        self.conversation_lock = Lock()
        logger.info(f"ConversationManager initialized with max_history={max_history_per_chat} per chat")
    
    def create_or_get_chat(self, chat_id: str) -> ChatSession:
        """Create a new chat session or get existing one."""
        # This method should only be called when the lock is already held
        if chat_id not in self.chat_sessions:
            chat_session = ChatSession(
                chat_id=chat_id,
                created_at=time.time(),
                last_activity=time.time(),
                turns=[]
            )
            self.chat_sessions[chat_id] = chat_session
            logger.info(f"Created new chat session {chat_id}")
        else:
            self.chat_sessions[chat_id].last_activity = time.time()
        
        return self.chat_sessions[chat_id]
    
    def add_turn(self, chat_id: str, user_message: str, assistant_response: str, sources: List[str] = None):
        """Add a conversation turn to a specific chat session."""
        try:
            # Use timeout to prevent blocking
            if self.conversation_lock.acquire(timeout=2.0):
                try:
                    chat_session = self.create_or_get_chat(chat_id)
                    
                    turn = ConversationTurn(
                        user_message=user_message,
                        assistant_response=assistant_response,
                        timestamp=time.time(),
                        sources=sources or []
                    )
                    
                    chat_session.turns.append(turn)
                    chat_session.last_activity = time.time()
                    
                    # Keep only recent history
                    if len(chat_session.turns) > self.max_history:
                        chat_session.turns = chat_session.turns[-self.max_history:]
                    
                    logger.info(f"Added conversation turn to chat {chat_id}. Total turns: {len(chat_session.turns)}")
                finally:
                    self.conversation_lock.release()
            else:
                logger.warning(f"Timeout acquiring lock for adding turn to chat {chat_id}")
        except Exception as e:
            logger.error(f"Error adding conversation turn to {chat_id}: {str(e)}")
    
    def get_history(self, chat_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get conversation history for a specific chat session."""
        with self.conversation_lock:
            if chat_id not in self.chat_sessions:
                return []
            
            chat_session = self.chat_sessions[chat_id]
            recent_turns = chat_session.turns[-max_turns:] if max_turns > 0 else chat_session.turns
            history = [
                {
                    "user": turn.user_message,
                    "assistant": turn.assistant_response
                }
                for turn in recent_turns
            ]
            
            logger.info(f"Retrieved {len(history)} conversation turns for chat {chat_id}")
            return history
    
    def clear_chat(self, chat_id: str):
        """Clear conversation history for a specific chat session."""
        with self.conversation_lock:
            if chat_id in self.chat_sessions:
                del self.chat_sessions[chat_id]
                logger.info(f"Cleared chat session {chat_id}")
    
    def _clear_chat_unsafe(self, chat_id: str):
        """Clear conversation history without acquiring lock (for internal use)."""
        if chat_id in self.chat_sessions:
            del self.chat_sessions[chat_id]
            logger.info(f"Cleared chat session {chat_id}")
    
    def get_all_chats(self) -> List[str]:
        """Get list of all active chat IDs."""
        with self.conversation_lock:
            return list(self.chat_sessions.keys())
    
    def cleanup_inactive_chats(self):
        """Remove chats that have been inactive for too long."""
        current_time = time.time()
        max_inactive_seconds = self.max_inactive_hours * 3600
        
        with self.conversation_lock:
            inactive_chats = []
            for chat_id, chat_session in self.chat_sessions.items():
                if current_time - chat_session.last_activity > max_inactive_seconds:
                    inactive_chats.append(chat_id)
            
            # Remove inactive chats without calling clear_chat to avoid deadlock
            for chat_id in inactive_chats:
                self._clear_chat_unsafe(chat_id)
            
            return len(inactive_chats)
    
    def get_chat_info(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific chat session."""
        with self.conversation_lock:
            if chat_id not in self.chat_sessions:
                return None
            
            chat_session = self.chat_sessions[chat_id]
            return {
                "chat_id": chat_session.chat_id,
                "created_at": chat_session.created_at,
                "last_activity": chat_session.last_activity,
                "total_turns": len(chat_session.turns)
            }

# Global variables
model_manager = None
rag_processor = None
conversation_manager = None
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
                
            question, chat_id, quantization, conversation_history, response_queue = request_data
            
            # Process the question
            result, status_code = rag_processor.process_question(question, 
                                                                 quantization, 
                                                                 conversation_history=conversation_history
                                                                 )
            
            # Send response first, then add to conversation history
            response_queue.put((result, status_code))
            
            # If successful, add to conversation history for this chat (async but non-blocking)
            if status_code == 200 and conversation_manager:
                def add_conversation_turn():
                    try:
                        answer = result.get('answer', '')
                        sources = result.get('sources', [])
                        conversation_manager.add_turn(chat_id, question, answer, sources)
                        logger.info(f"Added conversation turn to chat {chat_id}")
                    except Exception as e:
                        logger.error(f"Error adding conversation turn: {str(e)}")
                
                # Use a daemon thread for non-blocking conversation storage
                thread = threading.Thread(target=add_conversation_turn, daemon=True)
                thread.start()
            
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
        
        # Get chat_id and quantization from request
        chat_id = data.get('chat_id')
        quantization = data.get('quantization', None)
        
        # chat_id is required for proper conversation tracking
        if not chat_id:
            return jsonify({'error': 'chat_id is required for conversation tracking'}), 400
        
        # ✅ Récupérer l'historique stocké pour ce chat_id
        if conversation_manager:
            conversation_history = conversation_manager.get_history(chat_id, max_turns=10)
        else:
            conversation_history = []
        
        logger.info(f"Processing question for chat {chat_id} with {len(conversation_history)} stored turns")
            
        response_queue = Queue()
        request_queue.put((question, chat_id, quantization, conversation_history, response_queue))
        
        try:
            result, status_code = response_queue.get(timeout=ProductionConfig.REQUEST_TIMEOUT)
            
            # Add conversation context info to the response
            if status_code == 200:
                result['chat_id'] = chat_id
                result['conversation_history_used'] = len(conversation_history)
                
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

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get conversation history for a specific chat."""
    try:
        chat_id = request.args.get('chat_id')
        max_turns = int(request.args.get('max_turns', 10))
        
        if not chat_id:
            return jsonify({'error': 'chat_id is required'}), 400
        
        if not conversation_manager:
            return jsonify({'error': 'Conversation manager not initialized'}), 500
        
        history = conversation_manager.get_history(chat_id, max_turns)
        chat_info = conversation_manager.get_chat_info(chat_id)
        
        return jsonify({
            'chat_id': chat_id,
            'history': history,
            'total_turns': len(history),
            'chat_info': chat_info
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history for a specific chat."""
    try:
        data = request.json or {}
        chat_id = data.get('chat_id')
        
        if not chat_id:
            return jsonify({'error': 'chat_id is required'}), 400
        
        if not conversation_manager:
            return jsonify({'error': 'Conversation manager not initialized'}), 500
        
        conversation_manager.clear_chat(chat_id)
        return jsonify({
            'message': f'Chat history cleared for chat {chat_id}',
            'chat_id': chat_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chats/all', methods=['GET'])
def get_all_chats():
    """Get list of all active chat sessions."""
    try:
        if not conversation_manager:
            return jsonify({'error': 'Conversation manager not initialized'}), 500
        
        all_chat_ids = conversation_manager.get_all_chats()
        all_chats_info = []
        
        for chat_id in all_chat_ids:
            chat_info = conversation_manager.get_chat_info(chat_id)
            if chat_info:
                all_chats_info.append(chat_info)
        
        return jsonify({
            'chats': all_chats_info,
            'total_chats': len(all_chats_info)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting all chats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chats/cleanup', methods=['POST'])
def cleanup_inactive_chats():
    """Cleanup inactive chat sessions."""
    try:
        if not conversation_manager:
            return jsonify({'error': 'Conversation manager not initialized'}), 500
        
        cleaned_count = conversation_manager.cleanup_inactive_chats()
        return jsonify({
            'message': f'Cleaned up {cleaned_count} inactive chats',
            'cleaned_chats': cleaned_count
        }), 200
        
    except Exception as e:
        logger.error(f"Error cleaning up chats: {str(e)}")
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
    global model_manager, rag_processor, conversation_manager
    
    try:
        from config import ProductionConfig
        
        # Initialize conversation manager for chat sessions
        conversation_manager = ConversationManager(max_history_per_chat=20, max_inactive_hours=24)
        logger.info("Conversation manager initialized for chat sessions")
        
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
        
        logger.info("Application initialized successfully with conversation history support")
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