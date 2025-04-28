import os
import pickle
import torch
import logging
from datasets import Dataset
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    TrainerCallback,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase, # For type hinting
    )
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel, # For type hinting
)
from trl import SFTTrainer
from typing import Optional, List, Dict, Any, Tuple

# Use absolute import from project root
try:
    from llm_finetuning.training import model_loader as ml
except ImportError:
    logging.error("Failed to import model_loader.py. Make sure it's in the llm_finetuning/training directory and the script is run from the project root.")
    # Define dummy functions or exit if critical
    class ml:
        @staticmethod
        def model_caller(*args, **kwargs): raise NotImplementedError("model_loader.py not found")
    # exit(1) # Or raise error

# --- Constants ---
# File Paths (relative to project root)
# !! IMPORTANT: Consider making paths configurable !!
TRAIN_DATA_PATH = "data/wiki_train_data.pkl" # Assuming data is in data/ dir
EVAL_DATA_PATH = "data/wiki_validation_data.pkl"  # Assuming data is in data/ dir
OUTPUT_DIR = "llm_finetuning/training/finetuning_checkpoints2" # Checkpoint directory within training folder
FINAL_MODEL_SAVE_PATH = "llm_finetuning/training/fine_tuned_model" # Final PEFT adapter save path within training folder

# Data Sampling
TRAIN_SAMPLE_SIZE = 100_000
EVAL_SAMPLE_SIZE = 10_000

# Training Arguments
TRAIN_EPOCHS = 3
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 10
TRAIN_BATCH_SIZE = 12
EVAL_BATCH_SIZE = 12
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1.5e-4
OPTIMIZER = "paged_adamw_32bit"
LR_SCHEDULER = "cosine"
FP16_ENABLED = True # Use mixed precision
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
MAX_STEPS = -1 # -1 means based on epochs
GROUP_BY_LENGTH = True
REPORT_TO = "tensorboard" # Or "wandb", "none", etc.
LOAD_BEST_AT_END = True
METRIC_FOR_BEST = "eval_loss"
GREATER_IS_BETTER = False

# LoRA Config
LORA_R = 16
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"
# Target modules might vary based on the base model architecture
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "down_proj", "up_proj", "lm_head",
]

# Callbacks
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.001
PRINT_EXAMPLES_FREQ = 500 # Frequency to print examples in callback

# Dataset Formatting
DATASET_FORMAT_TEMPLATE = "<|system|>\n{Prompt}{eos_token}<|user|>\n{Train}{eos_token}<|assistant|>\n{Target}"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(
    train_path: str = TRAIN_DATA_PATH,
    eval_path: str = EVAL_DATA_PATH,
    train_sample: Optional[int] = TRAIN_SAMPLE_SIZE,
    eval_sample: Optional[int] = EVAL_SAMPLE_SIZE,
    seed: int = 42
) -> Optional[Tuple[Dataset, Dataset]]:
    """
    Loads training and evaluation data from pickle files, shuffles, and samples them.

    Args:
        train_path (str): Path to the training data pickle file.
        eval_path (str): Path to the evaluation data pickle file.
        train_sample (Optional[int]): Number of samples for training data. None for full dataset.
        eval_sample (Optional[int]): Number of samples for evaluation data. None for full dataset.
        seed (int): Random seed for shuffling.

    Returns:
        Optional[Tuple[Dataset, Dataset]]: A tuple (train_dataset, eval_dataset), or None on error.
    """
    try:
        logging.info(f"Loading training data from: {train_path}")
        with open(train_path, "rb") as f:
            train_data = pickle.load(f)
        logging.info(f"Loading evaluation data from: {eval_path}")
        with open(eval_path, "rb") as f:
            eval_data = pickle.load(f)

        # Ensure they are Dataset objects (or convert if necessary)
        if not isinstance(train_data, Dataset) or not isinstance(eval_data, Dataset):
             logging.error("Loaded data is not in expected Hugging Face Dataset format.")
             # Attempt conversion or return None
             return None

        logging.info("Shuffling datasets...")
        train_data = train_data.shuffle(seed=seed)
        eval_data = eval_data.shuffle(seed=seed)

        if train_sample is not None and train_sample > 0:
            logging.info(f"Sampling {train_sample} examples from training data.")
            train_data = train_data.select(range(min(train_sample, len(train_data))))
        if eval_sample is not None and eval_sample > 0:
            logging.info(f"Sampling {eval_sample} examples from evaluation data.")
            eval_data = eval_data.select(range(min(eval_sample, len(eval_data))))

        logging.info(f"Loaded {len(train_data)} training samples and {len(eval_data)} evaluation samples.")
        return train_data, eval_data

    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}. Please check paths.")
        return None
    except (pickle.UnpicklingError, IOError, EOFError) as e:
        logging.error(f"Error loading or unpickling data: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        return None


def create_training_args() -> TrainingArguments:
    """Creates and returns the TrainingArguments for the SFTTrainer."""
    logging.info(f"Creating TrainingArguments. Output directory: {OUTPUT_DIR}")
    train_params = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        # weight_decay=0.001, # Optional: Uncomment if needed
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        fp16=FP16_ENABLED,
        # bf16=False, # Optional: Use bf16 if hardware supports and fp16 is False
        max_steps=MAX_STEPS,
        group_by_length=GROUP_BY_LENGTH,
        report_to=REPORT_TO,
        load_best_model_at_end=LOAD_BEST_AT_END,
        metric_for_best_model=METRIC_FOR_BEST,
        greater_is_better=GREATER_IS_BETTER,
        save_total_limit=2, # Optional: Limit total number of checkpoints saved
    )
    return train_params


def create_lora_config() -> LoraConfig:
    """Creates and returns the LoraConfig for PEFT."""
    logging.info(f"Creating LoRA config. R={LORA_R}, Alpha={LORA_ALPHA}, Target Modules: {LORA_TARGET_MODULES}")
    peft_parameters = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=LORA_TASK_TYPE,
        target_modules=LORA_TARGET_MODULES,
    )
    return peft_parameters


class PrintExamplesCallback(TrainerCallback):
    """A custom callback that logs model predictions on test examples during training."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, test_samples: List[str], print_freq: int = PRINT_EXAMPLES_FREQ):
        self.tokenizer = tokenizer
        self.test_samples = test_samples
        self.model = model # Note: This might be the base model, not the PEFT model during training
        self.print_freq = print_freq
        logging.info(f"PrintExamplesCallback initialized. Will print {len(test_samples)} examples every {print_freq} steps.")
        logging.debug(f"Test samples provided to callback: {test_samples}")


    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Logs predictions at specified step intervals."""
        if state.global_step > 0 and state.global_step % self.print_freq == 0:
            logging.info(f"--- Evaluating examples at step {state.global_step} ---")
            # Ensure model is in eval mode and on the correct device
            # Accessing the potentially wrapped model via kwargs if available
            eval_model = kwargs.get('model', self.model) # Get the model passed by Trainer if possible
            eval_model.eval()
            device = eval_model.device # Get device from the model being evaluated

            with torch.no_grad():
                for i, sample in enumerate(self.test_samples):
                    try:
                        # Extract prompt part
                        parts = sample.split("<|assistant|>")
                        if len(parts) < 2:
                             logging.warning(f"Callback: Skipping sample {i} due to unexpected format: {sample[:100]}...")
                             continue
                        prompt_text = parts[0] + "<|assistant|>\n"
                        expected_response = parts[1].strip()

                        inputs = self.tokenizer(
                            prompt_text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=1024, # Use a reasonable max_length for input
                        ).to(device)

                        # Generate prediction
                        outputs = eval_model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=512, # Limit generated tokens for callback efficiency
                            eos_token_id=self.tokenizer.eos_token_id, # Ensure generation stops
                            pad_token_id=self.tokenizer.pad_token_id, # Set pad token id
                            do_sample=False, # Use greedy decoding for deterministic callback output
                        )
                        # Decode only the generated part
                        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                        logging.info(f"Callback Example {i+1} | Step {state.global_step}")
                        logging.info(f"  Input Prompt: {prompt_text}")
                        logging.info(f"  Generated:    {generated_text}")
                        logging.info(f"  Expected:     {expected_response}")

                    except Exception as e:
                        logging.error(f"Error during callback prediction for sample {i}: {e}", exc_info=True)

            # Ensure model is back in train mode
            eval_model.train()
            logging.info(f"--- Finished evaluating examples at step {state.global_step} ---")


def fine_tune_model(
    model: PeftModel, # Expecting the PEFT model here
    train_data: Dataset,
    eval_data: Dataset,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
    print_examples_callback: PrintExamplesCallback
) -> Optional[PeftModel]:
    """
    Fine-tunes the PEFT model using SFTTrainer.

    Args:
        model (PeftModel): The PEFT-wrapped model to fine-tune.
        train_data (Dataset): Training dataset.
        eval_data (Dataset): Evaluation dataset.
        training_args (TrainingArguments): Training arguments.
        tokenizer (PreTrainedTokenizerBase): Tokenizer.
        print_examples_callback (PrintExamplesCallback): Custom callback for logging examples.

    Returns:
        Optional[PeftModel]: The fine-tuned model, or None if training fails.
    """
    try:
        logging.info("Initializing SFTTrainer...")
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=model.peft_config.get("default"), # Pass LoRA config if needed by SFTTrainer version
            dataset_text_field="text", # Field containing the formatted text
            max_seq_length=1024, # Max sequence length for packing/truncation
            tokenizer=tokenizer,
            args=training_args,
            callbacks=[print_examples_callback, early_stopping_callback],
            # packing=True, # Optional: Pack sequences for efficiency if dataset is suitable
        )

        # Check for existing checkpoint and resume if found
        last_checkpoint = None
        output_dir = training_args.output_dir
        if os.path.isdir(output_dir):
             # Find the latest checkpoint directory (e.g., "checkpoint-1000")
             checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
             if checkpoints:
                  checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
                  last_checkpoint = os.path.join(output_dir, checkpoints[-1])
                  logging.info(f"Found existing checkpoint: {last_checkpoint}")

        logging.info("Starting model training...")
        if last_checkpoint:
            logging.info(f"Resuming training from checkpoint: {last_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            logging.info("No checkpoint found, starting training from scratch.")
            train_result = trainer.train()

        logging.info("Training finished.")
        logging.info(f"Train Result Metrics: {train_result.metrics}")

        # Save final metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logging.info("Saved training state and metrics.")

        # Note: SFTTrainer might automatically save the best model if load_best_model_at_end=True
        # The returned model should be the best one in that case.
        return trainer.model

    except Exception as e:
        logging.error(f"An error occurred during fine-tuning: {e}", exc_info=True)
        return None


def format_dataset_entry(entry: Dict[str, str], eos_token: str) -> str:
    """Formats a dataset entry using the predefined template."""
    try:
        # Ensure all keys are present in the entry dictionary
        return DATASET_FORMAT_TEMPLATE.format(
            Prompt=entry.get("Prompt", ""), # Use .get for safety
            Train=entry.get("Train", ""),
            Target=entry.get("Target", ""),
            eos_token=eos_token
        )
    except KeyError as e:
        logging.warning(f"Missing key {e} in dataset entry: {entry}. Returning empty string.")
        return ""
    except Exception as e:
        logging.error(f"Error formatting dataset entry: {e}. Entry: {entry}", exc_info=True)
        return ""


if __name__ == "__main__":
    logging.info("--- Starting Fine-Tuning Script ---")

    # --- 1. Load Base Model and Tokenizer ---
    logging.info("Loading base model and tokenizer using model_caller...")
    # model_caller prepares the model for k-bit training
    model_loaded, tokenizer_loaded = ml.model_caller(use_quantization=True) # Assuming quantization is desired
    if model_loaded is None or tokenizer_loaded is None:
        logging.critical("Failed to load base model or tokenizer. Exiting.")
        exit(1)
    logging.info("Base model and tokenizer loaded and prepared.")

    # --- 2. Create LoRA Config and Apply to Model ---
    lora_parameters = create_lora_config()
    try:
        logging.info("Applying LoRA PEFT adapter to the model...")
        # model_loaded should already be prepared by model_caller
        lora_model = get_peft_model(model_loaded, lora_parameters)
        logging.info("LoRA PEFT adapter applied successfully.")
        lora_model.print_trainable_parameters() # Log trainable parameters
    except Exception as e:
        logging.critical(f"Failed to apply PEFT model: {e}. Exiting.", exc_info=True)
        exit(1)

    # --- 3. Load and Prepare Data ---
    loaded_data = load_data()
    if loaded_data is None:
        logging.critical("Failed to load training/evaluation data. Exiting.")
        exit(1)
    train_data_raw, eval_data_raw = loaded_data

    logging.info("Formatting datasets...")
    try:
        train_transformed_list = [
            format_dataset_entry(entry, tokenizer_loaded.eos_token) for entry in train_data_raw
        ]
        eval_transformed_list = [
            format_dataset_entry(entry, tokenizer_loaded.eos_token) for entry in eval_data_raw
        ]
        # Filter out any empty strings resulting from formatting errors
        train_transformed_list = [s for s in train_transformed_list if s]
        eval_transformed_list = [s for s in eval_transformed_list if s]

        if not train_transformed_list or not eval_transformed_list:
             logging.critical("Dataset formatting resulted in empty lists. Check format_dataset_entry and input data.")
             exit(1)

        train_dataset = Dataset.from_dict({"text": train_transformed_list})
        eval_dataset = Dataset.from_dict({"text": eval_transformed_list})
        logging.info("Datasets formatted successfully.")
    except Exception as e:
        logging.critical(f"Error during dataset formatting: {e}. Exiting.", exc_info=True)
        exit(1)

    # --- 4. Setup Callbacks and Training Args ---
    training_arguments = create_training_args()

    # Select a few samples from the formatted eval dataset for the callback
    num_callback_samples = min(2, len(eval_dataset)) # Take up to 2 samples
    if num_callback_samples > 0:
         test_samples_for_callback = [eval_dataset["text"][i] for i in range(num_callback_samples)]
    else:
         test_samples_for_callback = []
         logging.warning("Evaluation dataset is empty, cannot create test samples for callback.")

    # Pass the base model to the callback for generation, as the PEFT model might behave differently during training steps
    print_examples_callback = PrintExamplesCallback(tokenizer_loaded, model_loaded, test_samples_for_callback)

    # --- 5. Fine-Tune the Model ---
    fine_tuned_model_result = fine_tune_model(
        lora_model, # Pass the PEFT model to the trainer
        train_dataset,
        eval_dataset,
        training_arguments,
        tokenizer_loaded,
        print_examples_callback,
    )

    # --- 6. Save the Final Model ---
    if fine_tuned_model_result:
        logging.info(f"Saving the final fine-tuned PEFT adapter to: {FINAL_MODEL_SAVE_PATH}")
        try:
            # SFTTrainer saves the full state, but we often just want the adapter
            # Use save_pretrained on the result model (which should be the PEFT model)
            fine_tuned_model_result.save_pretrained(FINAL_MODEL_SAVE_PATH)
            # Optionally save the tokenizer too
            tokenizer_loaded.save_pretrained(FINAL_MODEL_SAVE_PATH)
            logging.info("Final PEFT adapter and tokenizer saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save the final fine-tuned model: {e}", exc_info=True)
    else:
        logging.error("Fine-tuning failed or returned no model. Final model not saved.")

    logging.info("--- Fine-Tuning Script Finished ---")
