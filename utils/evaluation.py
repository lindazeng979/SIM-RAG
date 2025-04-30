from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import re
import string
import ast
import collections
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utils.text_processing import *
from collections import Counter

# ===================  EM Functions =================== 

# Function to normalize text: Lowercase, remove articles, punctuation, and extra spaces
def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(str(s)))))

# Function to compute exact match for multiple correct answers
def exact_match_multiple(prediction, ground_truths):
    """Returns 1 if the prediction matches any ground truth answer exactly, otherwise 0."""
    normalized_prediction = normalize_answer(prediction)
    
    # Ensure ground_truths is a list, even if it's a single string
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    for truth in ground_truths:
        normalized_truth = normalize_answer(truth)
        if normalized_prediction == normalized_truth:
            return 1
    return 0

#===================  Retrieval Functions =================== 

def extract_title_and_content(text):
    # Regex to match the title until the first newline and content after "Content: "
    title_pattern = re.compile(r'Title:\s*(.*)\n', re.DOTALL)
    content_pattern = re.compile(r'Content:\s*(.*)', re.DOTALL)
    
    # Extract title
    title_match = title_pattern.search(text)
    title = title_match.group(1).strip() if title_match else None
    
    # Extract content
    content_match = content_pattern.search(text)
    content = content_match.group(1).strip() if content_match else None
    
    return title, content

def exact_match_retrieval(prediction, ground_truths):
    title, content = extract_title_and_content(prediction)
    normalized_pred_title = normalize_answer(title)
    normalized_pred_text = normalize_answer(content)

    ret = [0, 0]
    for truth in ground_truths:
        normalized_title = normalize_answer(truth[0])
        if normalized_title == normalized_pred_title:
            ret[0] = 1
        if len(truth) > 1:
            normalized_text = normalize_answer(truth[1])
            if normalized_text == normalized_pred_text:
                ret[1] = 1
    return ret

#Precondition: Assumes df contains only the last turn for each datapoint
def get_f1_score(df):
    copy = df.copy()
    
    # Initialize variables to accumulate F1 scores
    total_f1 = 0
    total_predictions = len(copy)
    
    # Iterate through the last entries and compute F1 score for each
    for index, row in copy.iterrows():
        # Assuming 'Prediction' contains the model's predicted answer and 'Correct Answer' is a list of gold answers
        prediction = row['Reasoner Answer']
        gold_answers = safe_literal_eval(row['Gold Answer'])  # This should be a list of answers
        
        # Compute F1 score for the current prediction
        f1 = compute_f1_multiple(prediction, gold_answers)
        total_f1 += f1
    
    # Calculate average F1 score
    average_f1 = total_f1 / total_predictions if total_predictions > 0 else 0
    return average_f1

# =================== Generated data functions: get info/stats about generated data =================

def per_turn_generation(df):
    # Group the data by 'ID'
    grouped = df.groupby('ID')

    # Group by 'ID' and get the size of each group
    group_sizes = grouped.size()

    # Group by 'Turn' and count the occurrences of each 'Critic Output' value
    turn_counts = df.groupby('Turn')['Verdict'].value_counts().unstack(fill_value=0)

    # Print the results
    print("Verdict Distribution per Turn")
    print(turn_counts)

    # Calculate zero-shot EM for Turn = 0
    turn_0 = df[df['Turn'] == 0]
    total_turn_0 = len(turn_0)
    
    if total_turn_0 > 0:
        correct_turn_0 = turn_0['Verdict'].sum()  # Count of 1s in 'Critic Output'
        zero_shot_EM = correct_turn_0 / total_turn_0  # Calculate EM
        zero_shot_f1 =get_f1_score(turn_0)
        print(f"Zero Shot EM: {zero_shot_EM:.5f}%")
        print(f"Zero Shot F1:{zero_shot_f1:.5f}")
    else:
        print("No data to calculate Zero Shot EM.")
    return turn_counts,zero_shot_EM,zero_shot_f1

#Get EM from generated data
def get_EM_generation(df):
    last_entries = df.groupby('ID').tail(1)
    system_EM = accuracy_score(last_entries['Verdict'], [1] * len(last_entries))
    
    print(f"EM:{system_EM:.4f}")
    print(f"F1:{get_f1_score(last_entries):.4f}")
    return system_EM

# =================== System Functions ============================
def get_EM(df):
    last_entries = df.groupby('ID').tail(1)
    system_EM = accuracy_score(last_entries['Correct Answer'], [1] * len(last_entries))
    
    print(f"Total: {len(last_entries)}")
    print(f"EM: {system_EM:.5f}")
    print(f"F1: {get_f1_score(last_entries):.5f}")
    return system_EM

#Gives EM and F1 as well as critic statistics
def custom_evaluation(df, name = ''):
    last_entries = df.groupby('ID').tail(1)
    system_EM = accuracy_score(last_entries['Correct Answer'], [1] * len(last_entries))
    f1 = get_f1_score(last_entries)

    # Critic Accuracy, Precision, Recall, F1, False Positive Rate, True Negative Rate
    critic_accuracy = accuracy_score(df['Correct Answer'], df['Critic Output'])
    critic_precision = precision_score(df['Correct Answer'], df['Critic Output'])
    critic_recall = recall_score(df['Correct Answer'], df['Critic Output'])
    critic_f1 = f1_score(df['Correct Answer'], df['Critic Output'])
    
    tn, fp, fn, tp = confusion_matrix(df['Correct Answer'], df['Critic Output']).ravel()
    critic_false_positive_rate = fp / (fp + tn)
    critic_true_negative_rate = tn / (tn + fp)

    results = {
        'Num Data Points': len(df),
        '------------------------':'',
        'EM ': system_EM,
        'F1': f1,
        '-------------------------':'',
        'Critic Accuracy': critic_accuracy,
        'Critic Precision': critic_precision,
        'Critic Recall': critic_recall,
        'Critic F1 Score': critic_f1,
        'Critic False Positive Rate': critic_false_positive_rate,
        'Critic True Negative Rate': critic_true_negative_rate,
        '--------------------------':'',
    }
    
    # Print Results
    print()
    print()
    print(f"===== Evaluation Results =====")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        elif isinstance(value,int):
            print(f"{metric}: {value}")
        elif value is None:
            print(f"{metric}: No valid entries")
        else:
            print(f"{metric}")

    print("===================================================")
    return results

# =================== LLM redundancy functions ===========================
def measure_redundancy(df, overlap_threshold=20):
    total_groups = df["ID"].nunique()
    total_answers = 0
    total_rationales = 0
    redundant_answers = 0
    redundant_rationales = 0
    
    answer_redundancy_counts = Counter()  # Tracks how many groups have X repeated answers
    rationale_redundancy_counts = Counter()  # Tracks how many groups have X overlapping rationales
    
    # Group by "ID"
    grouped = df.groupby("ID")
    
    for group_id, group in grouped:
        #rint(f"\nProcessing ID: {group_id}")
        
        # 1) Count how many values in the group have the same "Reasoner Answer"
        answer_counts = group["Reasoner Answer"].value_counts()
        group_size = len(group)
        total_answers += group_size
        group_redundant_answers = sum(count for count in answer_counts if count > 1)
        redundant_answers += group_redundant_answers
        
        # Update the redundancy count for the group
        answer_redundancy_counts[group_redundant_answers] += 1
        
        #print(f"Group {group_id} - Redundant Answers: {group_redundant_answers}")
        
        # 2) Count how many "Reasoner Rationale" have overlapping words with other "Reasoner Rationale"
        rationale_texts = group["Reasoner Rationale"].dropna().tolist()
        total_rationales += len(rationale_texts)
        
        group_redundant_rationales = 0  # Count redundant rationales
        if len(rationale_texts) > 1:
            # Use CountVectorizer to get the word count for each rationale
            vectorizer = CountVectorizer().fit_transform(rationale_texts)
            vectors = vectorizer.toarray()
            
            # Compare each pair of rationales for word overlap
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    # Count overlapping words between two rationales
                    word_overlap = (vectors[i] & vectors[j]).sum()
                    
                    # If overlap exceeds the threshold, count it as redundant
                    if word_overlap >= overlap_threshold:
                        group_redundant_rationales += 1
                        break  # Only count the rationale once
            
            redundant_rationales += group_redundant_rationales
            rationale_redundancy_counts[group_redundant_rationales] += 1  # Update rationale redundancy count

        
        # Count this group as having 0 overlaps if no redundant rationales were found
        if group_redundant_rationales == 0:
            rationale_redundancy_counts[0] += 1

    # Calculate overall percentages
    answer_redundancy_percentage = (redundant_answers / total_answers) * 100 if total_answers > 0 else 0
    rationale_redundancy_percentage = (redundant_rationales / total_rationales) * 100 if total_rationales > 0 else 0
    
    print("\nOverall Redundancy:")
    print(f"Answer Redundancy: {answer_redundancy_percentage:.2f}%")
    print(f"Rationale Redundancy ({overlap_threshold} overlapping): {rationale_redundancy_percentage:.2f}%")
    
    def print_redundancy_counts(counts, label):
        print(f"\n{label} (How many groups have X repetitions/overlaps):")
        for key, value in sorted(counts.items(), reverse=True):
            print(f"  {value} group(s) with {key} repeated answer values/overlapping rationale(s)")

    print_redundancy_counts(answer_redundancy_counts, "Answer Redundancy Counts")
    print_redundancy_counts(rationale_redundancy_counts, f"Rationale Redundancy Counts ({overlap_threshold} overlapping)")
    
    return {
        "Overall Answer Redundancy (%)": answer_redundancy_percentage,
        "Overall Rationale Redundancy (%)": rationale_redundancy_percentage,
        "Answer Redundancy Counts": answer_redundancy_counts,
        "Rationale Redundancy Counts": rationale_redundancy_counts
    }


# =================== System Function ================================================================
def evaluate_per_turn(df):
    
    # Print the original distribution
    print("Original Distribution:")
    print(len(df))
    print(df['Critic Output'].value_counts())  # Total count of Critic Verdict 0 and 1
    print(df['Correct Answer'].value_counts()) 

    # Group the data by 'ID'
    grouped = df.groupby('ID')

    # Group by 'ID' and get the size of each group
    group_sizes = grouped.size()

    # Print the number of groups of each size
    print("\nGroup Size Distribution:")
    print(group_sizes.value_counts().sort_index())  # Count of groups by size

    # Group by 'Turn' and count the occurrences of each 'Critic Output' value
    turn_counts = df.groupby('Turn')['Critic Output'].value_counts().unstack(fill_value=0)

    # Print the results
    print("Critic Output counts per Turn:")
    print(turn_counts)

    # Group by 'Turn' and count the occurrences of each 'Critic Output' value
    turn_counts = df.groupby('Turn')['Correct Answer'].value_counts().unstack(fill_value=0)

    # Print the results
    print("Correct Answer counts per Turn:")
    print(turn_counts)

    # Calculate zero-shot EM for Turn = 0
    turn_0 = df[df['Turn'] == 0]
    total_turn_0 = len(turn_0)
    
    if total_turn_0 > 0:
        correct_turn_0 = turn_0['Correct Answer'].sum()  # Count of 1s in 'Critic Output'
        zero_shot_EM = correct_turn_0 / total_turn_0  # Calculate EM
        print(f"Zero Shot EM:{zero_shot_EM:.5f}")
        print(f"Zero Shot F1:{get_f1_score(turn_0):.5f}")
    else:
        print("No data to calculate Zero Shot EM.")



def compute_recall(df):
    # Initialize a dictionary to store recall for each id
    recall_by_id = {}
    # Check if 'Retrieved Titles' or 'Retrieved Title' exists and select the appropriate one
    retrieved_column = 'Retrieved Titles' if 'Retrieved Titles' in df.columns else 'Retrieved Title'
    # Check if 'Retrieved Titles' or 'Retrieved Title' exists and select the appropriate one
    gold_column = 'Gold Retrieved Docs' if 'Gold Retrieved Docs' in df.columns else 'Gold Retrieved Doc'

    # Group rows by 'id'
    grouped = df.groupby('ID')
    empty_set = 0 
    for idx, (id_, group) in enumerate(grouped):
        # Drop the last row of the group
        group = group.iloc[:-1]

        # Accumulate all retrieved titles for this group
        all_retrieved_titles = set()
        for titles in group[retrieved_column]:
            try:
                # Try to safely evaluate the string as a list
                titles = safe_literal_eval(titles)
                # If it's not a list, make it a list
                if not isinstance(titles, list):
                    titles = [titles]
            except (ValueError, SyntaxError):
                # If it can't be evaluated as a list, check if it's a single string
                if isinstance(titles, str):
                    titles = [titles]
            all_retrieved_titles.update(titles)

        # Get the gold retrieval documents
        gold_documents = set()
        for titles in group[gold_column]:
            try:
                # Try to safely evaluate the string as a list
                titles = safe_literal_eval(titles)
                # If it's not a list, make it a list
                if not isinstance(titles, list):
                    titles = [titles]
            except (ValueError, SyntaxError):
                # If it can't be evaluated as a list, check if it's a single string
                if isinstance(titles, str):
                    titles = [titles]
            gold_documents.update(titles)

        # Skip empty sets
        if not all_retrieved_titles or not gold_documents:
            empty_set = empty_set + 1
            continue

        # Calculate recall for this id group
        total_gold_docs = len(gold_documents)
        common_docs = gold_documents.intersection(all_retrieved_titles)

        if total_gold_docs > 0:
            recall_by_id[id_] = len(common_docs) / total_gold_docs
        else:
            recall_by_id[id_] = 0.0  # Handle edge case where there are no gold documents

    # Calculate the average recall across all IDs
    average_recall = sum(recall_by_id.values()) / len(recall_by_id) if recall_by_id else 0.0

    # Calculate the percentage of groups with no retrieval attempted
    no_retrieval = empty_set / len(grouped)

    # Get the number of groups with only 1 row
    num_single_row_groups = grouped.size()[grouped.size() == 1].count()

    # Print statistics and verify relationships
    print(f"Total data points ({len(grouped)}) = Total recall ({len(recall_by_id)}) + "
        f"(size of group size 1 ({num_single_row_groups} = no retrieval attempt ({empty_set})))")

    if len(grouped) != len(recall_by_id) + num_single_row_groups:
        print("Warning: Total data points do not equal the sum of total recall and size of group size 1!")

    if num_single_row_groups != empty_set:
        print("Warning: Size of group size 1 does not equal the number of no retrieval attempts!")

    # Print recall and percentage retrieval not attempted
    print(f"Recall score: {average_recall:.4f}")
    print(f"Percentage retrieval is not attempted: {no_retrieval:.4f}")

    # Return the average recall
    return average_recall, no_retrieval



# ===================  F1 functions==========================
def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()
  
# Compute F1 score for a prediction against multiple correct answers
def compute_f1_multiple(a_pred, gold_answers):
    pred_toks = get_tokens(a_pred)
    gold_toks_list = [get_tokens(ans) for ans in gold_answers]
    
    # Find the best F1 score across all gold answers
    best_f1 = 0
    for gold_toks in gold_toks_list:
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            best_f1 = max(best_f1, int(gold_toks == pred_toks))
        elif num_same != 0:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            best_f1 = max(best_f1, f1)
    
    return best_f1
