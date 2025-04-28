#Downsample 0s before Critic Training
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def main(input_path, train_path, test_path, log_path, split=0.2):
    # Load the CSV file
    df = pd.read_csv(input_path)
    df.rename(columns={'Verdict': 'Gate Verdict'}, inplace=True)

    with open(log_path, "a") as f:
        f.write(f"Generated Data Size: {len(df)}\n")
        generated_distribution = df['Gate Verdict'].value_counts().to_string()
        f.write("Generated Data Distribution:\n" + '\n'.join(generated_distribution.split('\n')[1:]) + "\n")
        f.write("====================================\n")

    # Separate rows with 'Gate Verdict == 0' and 'Gate Verdict == 1'
    df_verdict_0 = df[df['Gate Verdict'] == 0]
    df_verdict_1 = df[df['Gate Verdict'] == 1]

    # Balance the dataset by downsampling 'Gate Verdict == 0' to match the size of 'Gate Verdict == 1'
    if len(df_verdict_0) > len(df_verdict_1):
        df_verdict_0 = df_verdict_0.sample(n=len(df_verdict_1), random_state=42)

    # Combine the balanced dataset
    balanced_df = pd.concat([df_verdict_0, df_verdict_1])

    # Shuffle the balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Group the balanced data by 'ID' to keep groups together
    grouped_balanced = list(balanced_df.groupby('ID'))

    # Extract group IDs and perform stratified splitting based on 'Gate Verdict' of groups
    group_ids = [group_id for group_id, group in grouped_balanced]
    group_verdicts = [group['Gate Verdict'].mode()[0] for group_id, group in grouped_balanced]

    # Perform train-test split based on group IDs, ensuring that groups stay together
    train_ids, dev_ids = train_test_split(group_ids, test_size=split, random_state=42, stratify=group_verdicts)

    # Create train and dev DataFrames by selecting groups based on the split
    train_df = pd.concat([group for group_id, group in grouped_balanced if group_id in train_ids])
    dev_df = pd.concat([group for group_id, group in grouped_balanced if group_id in dev_ids])

    # Print final distribution
    # Print final distribution
    with open(log_path, "a") as f:
        f.write(f"Balanced Data Size: {len(balanced_df)}\n")
        
        balanced_distribution = balanced_df['Gate Verdict'].value_counts().to_string()
        f.write("Balanced Data Distribution:\n" + '\n'.join(balanced_distribution.split('\n')[1:]) + "\n")
        
        f.write(f"Train Data Size: {len(train_df)}\n")
        
        train_distribution = train_df['Gate Verdict'].value_counts().to_string()
        f.write("Train Data Distribution:\n" + '\n'.join(train_distribution.split('\n')[1:]) + "\n")
        
        f.write(f"Test Data Size: {len(dev_df)}\n")
        
        test_distribution = dev_df['Gate Verdict'].value_counts().to_string()
        f.write("Test Data Distribution:\n" + '\n'.join(test_distribution.split('\n')[1:]) + "\n")
        
        f.write("====================================\n")


    # Optionally, save the train and dev sets to CSV files
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Split the generated data into train and val data.')
    
    # Define command-line arguments
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment, used for naming and paths. This will generate default paths for input, train, test, and log files.')
    parser.add_argument('--input_path', type=str, required=False, help='Path to the input dataset. Default is generated using the experiment name.')
    parser.add_argument('--train_path', type=str, required=False, help='Path to the training dataset. Default is generated using the experiment name.')
    parser.add_argument('--test_path', type=str, required=False, help='Path to the testing dataset. Default is generated using the experiment name.')
    parser.add_argument('--log_path', type=str, required=False, help='Path to the log file. Default is generated using the experiment name.')
    parser.add_argument('--split', type=float, default=0.2, help='Proportion of the dataset to include in the test split (default is 0.2).')
        
    # Parse the arguments
    args = parser.parse_args()

        # Set default for input_path based on experiment_name if not provided
    if args.input_path is None:
        args.input_path = f'data/generated/{args.experiment_name}_generated.csv'
    if args.train_path is None:
        args.train_path = f'data/training/{args.experiment_name}_generated_train.csv'
    if args.test_path is None:
        args.test_path = f'data/training/{args.experiment_name}_generated_dev.csv'
    if args.log_path is None:
        args.log_path = f'logs/{args.experiment_name}_log.txt'

    # Call the run_system function with parsed arguments
    main(args.input_path, args.train_path, args.test_path, args.log_path, args.split)

# Example usage: python3 dm_training/prepare_training.py --dataset_name triviaqa
