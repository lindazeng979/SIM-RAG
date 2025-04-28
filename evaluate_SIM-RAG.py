    
import argparse
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_template import *
import re
import json
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import math
from accelerate import Accelerator
from utils.evaluation import *
from utils.text_processing import safe_literal_eval, extract_final_answer_and_rationale, extract_assistant_output, parse_query
import ast
import torch
import openai
    
def calculate_zeroshot(df):
     # Calculate zero-shot accuracy for Turn = 0

    turn_0 = df[df['Turn'] == 0]
    total_turn_0 = len(turn_0)
    
    if total_turn_0 > 0:
        correct_turn_0 = turn_0['Correct Answer'].sum()  
        zero_shot_accuracy = correct_turn_0 / total_turn_0  * 100
        f1 = get_f1_score(turn_0) * 100
        print(f"Zero Shot EM: {zero_shot_accuracy:.3f}")
        print(f"Zero Shot F1: {f1:.3f}")

def get_accuracy(df):
    last_entries = df.groupby('ID').tail(1)
    system_accuracy = accuracy_score(last_entries['Correct Answer'], [1] * len(last_entries)) * 100
    f1 = get_f1_score(last_entries) * 100

    print(f"SIM-RAG EM: {system_accuracy:.3f}")
    print(f"SIM-RAG F1: {f1:.3f}")
    print(f"Total Data Points: {len(last_entries)}")
    return system_accuracy

def evaluate_model(file_path):
    # Construct the file path for the predictions file
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Load the dataframe from the predictions file
    df = pd.read_csv(file_path)
    df_zs = df.groupby('ID').head(1)

    calculate_zeroshot(df)
    get_accuracy(df)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate the model's EM and F1 based on predictions.")
    
    # Add argument for the model and dataset name
    parser.add_argument('--experiment_name', type=str,
                        help="Experiment name, same as name input to running the system")

    # Parse the arguments
    args = parser.parse_args()
    file_path = f"predictions/{args.experiment_name}_predictions.csv"

    # Call the evaluate_model function with the input argument
    evaluate_model(file_path)

if __name__ == "__main__":
    main()
