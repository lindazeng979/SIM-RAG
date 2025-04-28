from datasets import load_dataset
import csv
import json

# Load the train and dev datasets
with open('data/2WikiMultiHopQA/data/train.json', 'r') as f:
    train_dataset = json.load(f)[0:10000]  # Load the first 10000 entries of the train dataset

with open('data/2WikiMultiHopQA/data/dev.json', 'r') as f:
    dev_dataset = json.load(f)  # Load the entire dev datasetprint(len(train_dataset))
print(len(train_dataset))
print(len(dev_dataset))

# Output CSV file paths
train_output_csv = '/data/diji/SIM-RAG/master/data/original/2wikimultihopqa_train.csv'
dev_output_csv = '/data/diji/SIM-RAG/master/data/original/2wikimultihopqa_dev.csv'

# Function to process and save a split to CSV
def save_split_to_csv(data_split, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['ID', 'Question', 'Answers', 'Documents'])

        # Process each entry in the split
        for i, entry in enumerate(data_split):
            # Extract fields
            qid = entry.get('_id')  # Unique ID (HotpotQA provides 'id' as question identifier)
            question = entry.get('question', 'N/A')
            
            # Extract the answers, which are a list of possible answers
            answers = [entry.get('answer', [])]
            
            # Process 'context' to extract document titles (if any)
            supporting = entry.get('supporting_facts', [])
            documents = [doc[0] for doc in supporting]
            
            # Write row to CSV
            writer.writerow([qid, question, str(answers), str(documents)])

# Save train and validation splits
save_split_to_csv(train_dataset, train_output_csv)  # First 10k entries for training
save_split_to_csv(dev_dataset, dev_output_csv)  # Full validation set

print(f"Train data saved to {train_output_csv}") #10k
print(f"Dev data saved to {dev_output_csv}") #12576
