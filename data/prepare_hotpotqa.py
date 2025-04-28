from datasets import load_dataset
import csv

# Load HotpotQA dataset from Hugging Face
train_dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train[0:10000]")  # Use the first 10,000 examples
dev_dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")  # Validation split
print(len(train_dataset))
print(len(dev_dataset))

# Output CSV file paths
train_output_csv = 'data/original/hotpotqa_train.csv'
dev_output_csv = 'data/original/hotpotqa_dev.csv'

# Function to process and save a split to CSV
def save_split_to_csv(data_split, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['ID', 'Question', 'Answers', 'Documents'])

        # Process each entry in the split
        for i, entry in enumerate(data_split):
            # Extract fields
            qid = entry.get('id')  # Unique ID (HotpotQA provides 'id' as question identifier)
            question = entry.get('question', 'N/A')
            
            # Extract the answers, which are a list of possible answers
            answers = [entry.get('answer', [])]
            
            # Process 'context' to extract document titles (if any)
            documents = entry.get('supporting_facts', []).get('title')
            
            # Write row to CSV
            writer.writerow([qid, question, str(answers), str(documents)])

# Save train and validation splits
save_split_to_csv(train_dataset, train_output_csv)  # First 10k entries for training
save_split_to_csv(dev_dataset, dev_output_csv)  # Full validation set

print(f"Train data saved to {train_output_csv}") #10k
print(f"Dev data saved to {dev_output_csv}") #7.4k
