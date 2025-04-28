from datasets import load_dataset
import csv
import json

# Load TriviaQA from Hugging Face
train_dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="train[0:10000]")
dev_dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia",split="validation")

# Output CSV file
train_output_csv = 'data/original/triviaqa_train.csv'
dev_output_csv = 'data/original/triviaqa_dev.csv'

# Function to process and save a split to CSV
def save_split_to_csv(data_split, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['ID', 'Question', 'Answers', 'Documents'])

        # Process each entry in the split
        for i, entry in enumerate(data_split):
            # Extract fields
            qid = entry.get('question_id')  # Unique ID
            question = entry.get('question', 'N/A')
            answers = entry.get('answer', {}).get('aliases', [])

            # Process 'entity_pages'
            entity_pages_raw = entry.get('entity_pages', [])
            documents = entity_pages_raw.get('title')

            # Write row to CSV
            writer.writerow([qid, question, str(answers), str(documents)])

# Save train and validation splits
save_split_to_csv(train_dataset, train_output_csv) #use first 10k
save_split_to_csv(dev_dataset, dev_output_csv) #8k

print(f"Train data saved to {train_output_csv}") #10k
print(f"Dev data saved to {dev_output_csv}") #8k
