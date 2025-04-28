#This is the main Critic training script.
import argparse
import config
from data_processing import load_data, prepare_dataset, create_dataset_dict, tokenize_data, get_data_collator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune or evaluate a FLAN-T5 model for binary classification.")
    parser.add_argument("--model_path", type=str, default=config.MODEL_PATH, help="Name or path of the pre-trained model")
    parser.add_argument("--max_source_length", type=int, default=config.MAX_SOURCE_LENGTH, help="Maximum source sequence length")
    parser.add_argument("--max_target_length", type=int, default=config.MAX_TARGET_LENGTH, help="Maximum target sequence length")
    parser.add_argument("--experiment_name", type=str, default=config.EXPERIMENT_NAME, help="Name of the experiment, used for naming and paths.")
    parser.add_argument("--train_path", type=str, help="Path to the training data. Default is generated using the experiment name.")
    parser.add_argument("--val_path", type=str, help="Path to the validation data. Default is generated using the experiment name.")
    parser.add_argument("--test_path", type=str, help="Path to the test data. Default is generated using the experiment name.")
    parser.add_argument("--log_path", type=str, help="Path to the log file. Default is generated using the experiment name.")
    parser.add_argument("--output_dir", type=str, help="Output directory for the saved model. Default is generated using the experiment name.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=config.PER_DEVICE_TRAIN_BATCH_SIZE, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=config.PER_DEVICE_EVAL_BATCH_SIZE, help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--num_train_epochs", type=int, default=config.NUM_TRAIN_EPOCHS, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=config.WARMUP_STEPS, help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=config.LOGGING_STEPS, help="Number of steps between logging")
    parser.add_argument("--eval_strategy", type=str, default=config.EVAL_STRATEGY, help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default=config.SAVE_STRATEGY, help="Save strategy")
    parser.add_argument("--save_total_limit", type=int, default=config.SAVE_TOTAL_LIMIT, help="Maximum number of checkpoints to keep")
    parser.add_argument("--fp16", action="store_true", default=config.FP16, help="Use mixed precision training")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate the model without training")
    parser.add_argument('--weighted_training', action="store_true", default=False, help='Boolean flag to indicate whether to use weighted training.')
    return parser.parse_args()

# for weighted training
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Only select logits for '0' and '1' tokens
        relevant_logits = logits[:, :, [self.tokenizer.encode('0', add_special_tokens=False)[0], 
                                        self.tokenizer.encode('1', add_special_tokens=False)[0]]]
    
        new_labels = torch.full_like(labels, -100) 
        new_labels[labels == self.tokenizer.encode('0', add_special_tokens=False)[0]] = 0  
        new_labels[labels == self.tokenizer.encode('1', add_special_tokens=False)[0]] = 1 

        # Set weights: higher cost for predicting 1 (accept)
        w_reject = 1.0  # Weight for predicting 0
        w_accept = 1.0  # Weight for predicting 1
        
        # Create weighted loss function
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([w_reject, w_accept]).to(logits.device),
            ignore_index=-100  # Keep ignoring padding tokens
        )
        
        # Compute loss
        # shifted_logits = logits[..., :-1, :].contiguous() # for real seq2seq tasks
        # shifted_labels = labels[..., 1:].contiguous()
        loss = loss_fct(relevant_logits.view(-1, 2), new_labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    

def load_model_and_tokenizer(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        print("Cannot load tokenizer from the finetuned model path, loading from the subfolder.")
        tokenizer = AutoTokenizer.from_pretrained(model_path + "/_tokenizer")
    return model, tokenizer

def compute_metrics(tokenizer):
    def compute(eval_prediction):
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            decoded_labels, 
            decoded_preds, 
            average=None,
            labels=['0', '1']
        )
        
        return {
            'accuracy': accuracy_score(decoded_labels, decoded_preds),
            'precision_0': precision[0],
            'precision_1': precision[1],
            'recall_0': recall[0],
            'recall_1': recall[1],
            'f1_0': f1[0],
            'f1_1': f1[1]
        }
    return compute
    
def train_model(model, tokenizer, tokenized_dataset, data_collator, args):
    training_args = Seq2SeqTrainingArguments(
        report_to=None,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        predict_with_generate=True,
        load_best_model_at_end=True, 
        metric_for_best_model="accuracy",
    )

    if args.weighted_training:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            compute_metrics=compute_metrics(tokenizer),
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            compute_metrics=compute_metrics(tokenizer),
        )

    trainer.train()
    return trainer

def evaluate_model(trainer, tokenized_dataset):
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset["val"])
    return metrics

def save_model(trainer, tokenizer, output_dir):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir + "_tokenizer")

def main():
    # os.environ['WANDB_DISABLED'] = "true"
    print("Visible CUDA Devices:", os.environ['CUDA_VISIBLE_DEVICES'])
    args = parse_args()

    if (args.train_path is None):
        args.train_path = f"data/training/{args.experiment_name}_generated_train.csv"
    if (args.val_path is None):
        args.val_path = f"data/training/{args.experiment_name}_generated_dev.csv"
    if (args.test_path is None):
        args.test_path = f"data/training/{args.experiment_name}_generated_dev.csv"
    if (args.output_dir is None):
        args.output_dir = f"dm_training/finetuned_checkpoints/{args.experiment_name}"
    if (args.log_path is None):
        args.log_path = f"logs/{args.experiment_name}_log.txt"

    # Load and prepare data
    raw_train, raw_val, raw_test = load_data(args.train_path, args.val_path, args.test_path)
    train_set = prepare_dataset(raw_train)
    val_set = prepare_dataset(raw_val)
    test_set = prepare_dataset(raw_test)
    dataset = create_dataset_dict(train_set, val_set, test_set)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Tokenize data
    tokenized_dataset = tokenize_data(dataset, tokenizer, args.max_source_length, args.max_target_length)

    # Prepare data collator
    data_collator = get_data_collator(tokenizer, model)
    
    if args.eval_only:
        # Only evaluate the model
        eval_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            predict_with_generate=True,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics(tokenizer),
        )
        metrics = trainer.evaluate(eval_dataset=tokenized_dataset["val"])
        print(f"Evaluation metrics: {metrics}")
    else:
        # Train and evaluate model
        trainer = train_model(model, tokenizer, tokenized_dataset, data_collator, args)
        print("Evaluating Model:")
        metrics = evaluate_model(trainer, tokenized_dataset)
        print(f"Evaluation metrics: {metrics}")
        with open(args.log_path,'a') as f:
            f.write(f"Critic Evaluation Metrics: {metrics}\n")
            f.write("========================================\n")
        save_model(trainer, tokenizer, args.output_dir)
        print("Model Saved")

if __name__ == "__main__":
    main()