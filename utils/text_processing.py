#this is used for all the parsing code during generation and inference
import re
import ast
import json

# Gets the value from dataframe that is intended to be a list. if its a string, parses it to be list. otherwise returns string or whatever type it is if not string or list.
def safe_literal_eval(val):
    if isinstance(val, list):
        return val  # Already a list, return as is
    elif isinstance(val, str):
        # Remove extra quotes around the string if present
        val = val.strip('"').strip("'")
        try:
            # Attempt to parse the string
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val  # If parsing fails, return the string as is
    return val  # Return the value if it's not a list or string

# Parses what the LLM responds from batching code and removes extra empty lines between 'Answer' and 'Rationale'
def extract_assistant_output(text):
    # Define the regex pattern to capture everything after 'assistant'
    pattern = r"assistant\n\n(.*)$"

    # Use re.search to find the match
    match = re.search(pattern, text, re.DOTALL)

    # If a match is found, process the response
    if match:
        output = match.group(1).strip()
        # Remove any extra empty lines between 'Answer' and 'Rationale'
        output = re.sub(r"\n\s*\n", "\n", output)
        return output
    else:
        return "Output not found"


def extract_number(text):
    # Define the regex pattern to capture only numbers (integers or decimals)
    pattern = r'answer is [^\d]*([\d,]+(?:\.\d+)?)'
    match = re.search(pattern, text)
    
    if match:
        # Remove commas from the captured number and return it
        number = match.group(1).replace(',', '')
        return number
    
    # If no "The answer is" pattern found, extract the last number in the text
    last_number_match = re.findall(r'[\d,]+(?:\.\d+)?', text)
    
    if last_number_match:
        # Get the last match and remove commas
        last_number = last_number_match[-1].replace(',', '')
        return last_number
    
    return "Error: No number found"

def extract_final_answer(text, question_type="OEQ"):
    if question_type =="MATH":
        return extract_number(text)
    else:
        # from "Answer: ... Rationale: ..."
        # Define the regex pattern to capture the answer
        pattern = r"Answer:\s*(.*?)\s*Rationale:"

        # Use re.search to find the match
        match = re.search(pattern, text)

        # Extract and return the answer
        if match:
            return match.group(1).strip()
        else:
            return "Answer not found"


# parses within an llm response to give answer, rationale
def extract_final_answer_and_rationale(text, question_type="OEQ"):
    if question_type == "MATH":
        return extract_number(text), text
    else:
        # Define the regex pattern to capture both the answer and rationale
        pattern = r"Answer:\s*(.*?)\s*Rationale:\s*(.*)"

        # Use re.search to find the match
        match = re.search(pattern, text)

        # Extract and return the answer and rationale as a tuple
        if match:
            answer = match.group(1).strip()
            rationale = match.group(2).strip()
            return answer, rationale
        else:
            return "Answer not found", "Rationale not found"

# parses search query from the json output of llm
def parse_query(response):
    # Pattern to match JSON containing "search_query" with new lines and whitespace
    search_query_pattern = re.compile(r'\{\s*"search_query":\s*".*?"\s*\}', re.DOTALL)

    # Search for the JSON-like part in the response
    match = search_query_pattern.search(response)
    try:
        if match:
            json_str = match.group(0)
            json_response = json.loads(json_str)
            return json_response.get("search_query")
        else:
            # Attempt to salvage by extracting text between 'search_query' and 'reasoning'
            salvage_pattern = re.compile(r'search_query.*?:(.*?)[,\n\r]+.*?reasoning', re.DOTALL)
            salvage_match = salvage_pattern.search(response)
            if salvage_match:
                # Clean the extracted text to remove any unwanted characters
                return salvage_match.group(1).strip().strip('"')
            else:
                print(f"Attempt failed: 'search_query' and 'reasoning' key not found in response.")
                return response
    except json.JSONDecodeError:
        print(f"Attempt failed: json decode error")
        return response
    except Exception as e:
        print(f"Attempt failed: unknown exception - {str(e)}")
        return response

    return response