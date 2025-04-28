# FLAN-T5 Fine-tuning for Binary Classification

This project fine-tunes a FLAN-T5 model for binary classification tasks, specifically for determining whether an answer should be accepted or rejected based on given context and rationale.

## Project Structure

- `prepare_training.py`: Script to generate train and dev data before training.
- `main.py`: Main script for training and evaluating the model
- `config.py`: Configuration file with default settings for model and training
- `data_processing.py`: Utility functions for data loading, preprocessing, and tokenization

## Setup

1. Install required dependencies:
   ```
   pip install transformers datasets pandas numpy scikit-learn torch
   ```

2. Adjust settings in `config.py` as needed, particularly data paths and model parameters

## Usage

Run the training process:
```
python3 main.py
```

This will use the default settings from `config.py`.

To override default settings, use command-line arguments:
```
python3 main.py --experiment_name name
```

Or, if you have custom paths:

```
python3 main.py --/path/to/train_data.csv --val_path /path/to/val_data.csv --output_dir /path/to/output_dir
```

For a full list of available options, run:

```
python3 main.py --help
``` 

## Environment-Specific Notes

To run the script on specific GPUs, follow these steps:

1. Set the `CUDA_VISIBLE_DEVICES` global environment variable to the desired GPU IDs.
2. Use the `nohup` command to run the script in the background.
3. Redirect the output to a log file.

Example command to run the script on GPU 2 and GPU 3, with logs saved in `training_logs/`:
```
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,3 python3 main.py' > training_logs/log.txt 2>&1 &
```

## Eval-only Model

To run the evaluation process only, use the following command:
```
python3 main.py --eval_only --model_path /path/to/saved_checkpoint
```

## Notes on Data Format

- The script assumes that the input data is in CSV format with columns 'Reasoner Task Content,' 'Reasoner Answer,' 'Reasoner Rationale,' and 'Gate Verdict'
- The 'Gate Verdict' column should contain binary values (0 or 1) indicating whether the answer should be accepted (1) or rejected (0)