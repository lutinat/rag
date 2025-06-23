import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm
import random
from datetime import datetime
import wandb
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

class SatlantisQADataset(Dataset):
    """Dataset class for Satlantis QA pairs with simple question-answer format."""
    
    def __init__(self, jsonl_path, tokenizer, max_length=2048, split_ratio=0.9, mode='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load and process data
        with open(jsonl_path, 'r') as f:
            all_data = [json.loads(line) for line in f]
        
        # Split data
        random.shuffle(all_data)
        split_idx = int(len(all_data) * split_ratio)
        
        if mode == 'train':
            self.raw_data = all_data[:split_idx]
        else:  # validation
            self.raw_data = all_data[split_idx:]
        
        # Process data into simple question-answer format
        self.process_data()
        
    def process_data(self):
        """Process raw data into simple question-answer pairs."""
        
        for item in self.raw_data:
            question = item['question']
            answer = item['answer']
            
            # Simple format: question as input, answer as expected output
            formatted_text = f"{question}\n{answer}"
            self.data.append(formatted_text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize with special tokens
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class SatlantisModelFineTuner:
    """Fine-tuning class for creating a general-purpose Satlantis model."""
    
    def __init__(self, 
                 base_model_name="microsoft/Phi-4-mini-instruct",
                 data_path="data_processing/generated_questions.jsonl",
                 output_dir="models/satlantis-general-v1",
                 use_lora=True,
                 quantization="4bit"):
        
        self.base_model_name = base_model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.quantization = quantization
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup_tokenizer(self):
        """Setup tokenizer with proper padding token."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        print(f"Tokenizer loaded: {self.base_model_name}")
        
    def setup_model(self):
        """Setup model with optional quantization and LoRA."""
        
        # Configure quantization
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        
        # Resize token embeddings if needed
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup LoRA if requested
        if self.use_lora:
            if quantization_config:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA configuration for general-purpose fine-tuning
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # Rank
                lora_alpha=32,  # Alpha parameter
                lora_dropout=0.1,  # Dropout
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        print(f"Model loaded: {self.base_model_name}")
        
    def setup_datasets(self):
        """Setup training and validation datasets."""
        self.train_dataset = SatlantisQADataset(
            self.data_path, 
            self.tokenizer, 
            mode='train'
        )
        
        self.val_dataset = SatlantisQADataset(
            self.data_path, 
            self.tokenizer, 
            mode='val'
        )
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        
    def setup_training_args(self):
        """Setup training arguments optimized for general-purpose model."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"satlantis-general-{timestamp}"
        
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            
            # Training schedule
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            
            # Optimization
            learning_rate=2e-4,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            
            # Evaluation and saving
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # Optimization settings
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=True,
            
            # Logging
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,
            report_to="wandb" if wandb.run else None,
            run_name=run_name,
            
            # Misc
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
        )
    
    def train(self):
        """Main training function."""
        
        print("🚀 Starting Satlantis General Model Fine-tuning")
        
        # Setup all components
        self.setup_tokenizer()
        self.setup_model()
        self.setup_datasets()
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        print("🔥 Training started...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"✅ Training completed! Model saved to {self.output_dir}")
        
        # Save training info
        self.save_training_info()
        
    def save_training_info(self):
        """Save training configuration and info."""
        info = {
            "base_model": self.base_model_name,
            "data_path": self.data_path,
            "training_timestamp": datetime.now().isoformat(),
            "use_lora": self.use_lora,
            "quantization": self.quantization,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
                         "purpose": "Simple question-answering model fine-tuned on Satlantis data",
             "recommended_use_cases": [
                 "Question answering about Satlantis technologies",
                 "Space technology explanations",
                 "Information about satellite systems",
                 "Technical support and documentation"
             ]
        }
        
        with open(os.path.join(self.output_dir, "training_info.json"), 'w') as f:
            json.dump(info, f, indent=2)



def main():
    """Main function to run fine-tuning."""
    
    # Initialize wandb (optional)
    try:
        wandb.init(
            project="satlantis-general-model",
            name=f"finetune-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": "microsoft/Phi-4-mini-instruct",
                "task": "general-purpose-satlantis-model",
                "approach": "LoRA + 4bit quantization"
            }
        )
    except:
        print("⚠️ Wandb not available, continuing without logging")
    
    # Create fine-tuner and start training
    fine_tuner = SatlantisModelFineTuner(
        base_model_name="microsoft/Phi-4-mini-instruct",
        data_path="data_processing/generated_questions.jsonl",
        output_dir="models/satlantis-general-v1",
        use_lora=True,
        quantization="4bit"
    )
    
    fine_tuner.train()
    
    print("🎉 All done! Your Satlantis model is ready!")

if __name__ == "__main__":
    main() 