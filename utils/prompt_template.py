# Formatting Prompts
formatting = '''The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{
 "search_query": string  // The search query, as a unique string.
 "reasoning": string  // Any reasoning or notes or comments you have, as a unique string.
}  
'''

formatting2 = '''Respond only with the following format, nothing else:
Answer: [Provide the answer here]
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
'''


formatting_abstain = '''Respond only with the following format, nothing else:
Answer: [Provide the answer here or 'unsure']
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
'''


force_answer_prompt = (
    "You are a knowledgeable question-answering assistant. Based on the context provided (if any), answer the following "
    "multihop question. Provide a brief multihop explanation. If you do not know part of what is referenced in the question, "
    "do not try to make it up. If unsure, respond with 'unsure.' Provide your answer along with a brief explanation of your reasoning."
) + formatting2


#============================================
search_query_prompt = (
    "You are tasked with answering a complex, multihop question using a similarity search engine for Wikipedia. "
    "To tackle this, you need to break down the question into simpler parts. Begin by generating a search query "
    "to find information about one of the key components or entities involved in the question. IMPORTANT: avoid repeating search queries that already appear in the context or searching for documents that already are in the context."
    "IMPORTANT: Some context documents are wrong so do not keep searching for them. Make sure your query "
    "is focused just on identifying or understanding this singular part or entity, without delving into how it connects "
    "to the overall question. Please provide only the search query with no additional context or explanations. "
) + formatting

search_query_prompt_multi_ex = (
    "You are tasked with answering a complex, multihop question using a similarity search engine for Wikipedia. "
    "To tackle this, you need to break down the question into simpler parts. Begin by generating a search query "
    "to find information about one of the key components or entities involved in the question. IMPORTANT: avoid repeating search queries that already appear in the context or searching for documents that already are in the context."
    "IMPORTANT: Some context documents are wrong so do not keep searching for them. Make sure your query "
    "is focused just on identifying or understanding this singular part or entity, without delving into how it connects "
    "to the overall question. "
    "Here are some examples of successful search queries and their desired functionality:"
    '''Question: What year saw the creation of the region where the county of Hertfordshire is located?
Query: Hertfordshire
Retrieved Document: Title: Hertfordshire Content: Hertfordshire is a ceremonial county in the East of England and one of the home counties. It borders Bedfordshire to the north-west, Cambridgeshire to the north-east, Essex to the east, Greater London to the south and Buckinghamshire to the west. The largest settlement is Watford, and the county town is Hertford.
Query: East of England
Retrieved document: Title: East of England Content: The East of England is one of nine official regions of England at the first level of NUTS for statistical purposes. It was created in 1994 and was adopted for statistics from 1999. It includes the ceremonial counties of Bedfordshire, Cambridgeshire, Essex, Hertfordshire, Norfolk and Suffolk. Essex has the highest population in the region.
Answer: 1994

Question: Gunmen from Laredo starred which narrator of "Frontier"?
Query: Gunmen from Laredo
Retrieved Document: Title: Gunmen from Laredo Content: Gunmen from Laredo is a 1959 American western film produced and directed by Wallace MacDonald, which stars Robert Knapp, Maureen Hingert, and Walter Coy.
Query: Walter Coy
Retrieved Document: Title: Walter Darwin Coy Content: Walter Darwin Coy (January 31, 1909 – December 11, 1974) was an American stage, radio, film, and, principally, television actor, originally from Great Falls, Montana. He was best known for narrating the NBC western anthology series, "Frontier", which aired early Sunday evenings in the 1955–1956 season.
Answer: Walter Darwin Coy

Please provide only the search query with no additional context or explanations.
'''
) + formatting

search_query_prompt_2wiki = (
    "You are tasked with answering a complex, multihop question using a similarity search engine for Wikipedia. "
    "To tackle this, you need to break down the question into simpler parts. Begin by generating a search query "
    "to find information about one of the key components or entities involved in the question. IMPORTANT: avoid repeating search queries that already appear in the context or searching for documents that already are in the context."
    "IMPORTANT: Some context documents are wrong so do not keep searching for them. Make sure your query "
    "is focused just on identifying or understanding this singular part or entity, without delving into how it connects "
    "to the overall question. "
    "Here are some examples of successful search queries and their desired functionality:"
    '''Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
Query: Theodor Haecker
Retrieved Document: Title: Theodor Haecker Content: Theodor Haecker (June 4, 1879 - April 9, 1945) was a German writer, translator and cultural critic.
Query: Harry Vaughan Watkins
Retrieved Document: Title: Harry Vaughan Watkins Content: Harry Vaughan Watkins (10 September 1875 – 16 May 1945) was a Welsh rugby union player.
Answer: Harry Vaughan Watkins

Question: Where did the founder of Versus die?
Query: Versus
Retrieved Document: Title: Versus (Versace) Content: Versus (Versace) was a diffusion line of the Italian luxury fashion house Versace. It was founded in 1989 by Gianni Versace as a gift to his sister Donatella.
Query: Gianni Versace
Retrieved Document: Title: Gianni Versace Content: Giovanni Maria "Gianni" Versace (2 December 1946 – 15 July 1997) was an Italian fashion designer, socialite and businessman. On 15 July 1997, he was murdered outside his Miami Beach mansion, Casa Casuarina, by serial killer Andrew Cunanan.
Answer: Miami Beach

Question: Who is the grandchild of Dambar Shah?
Query: Dambar Shah
Retrieved Document: Title: Dambar Shah Content: Dambar Shah (?–1645) was a King of the Gorkha Kingdom, present-day Gorkha District, Nepal who reigned from 1636 until his death in 1645. He was the father of Krishna Shah.
Query: Krishna Shah
Retrieved Document: Title: Krishna Shah (Nepalese royal) Content: Krishna Shah (?–1661) was King of the Gorkha Kingdom in the Indian subcontinent, present-day Nepal, who succeeded his father Dambar Shah and reigned from 1645 till his death in 1661. He was the father of his successor Rudra Shah.
Answer: Rudra Shah

Question: Are both director of film FAQ: Frequently Asked Questions and director of film The Big Money from the same country?
Query: FAQ: Frequently Asked Questions
Retrieved Document: Title: FAQ: Frequently Asked Questions Content: is a feature-length dystopian movie, written and directed by Carlos Atanes and released in 2004.
Query: The Big Money film
Retrieved Document: Title: The Big Money (film) Content: The Big Money is a 1958 British comedy film directed by John Paddy Carstairs and starring Ian Carmichael, Belinda Lee and Kathleen Harrison.
Query: Carlos Atanes
Retrieved Document: Title: Carlos Atanes Content: Carlos Atanes is a Spanish film director, writer and playwright.
Query: John Paddy Carstairs
Retrieved Document: Title: John Paddy Carstairs Content: John Paddy Carstairs was a British film director and television director, usually of light-hearted subject matter. 
Answer: No


Please provide only the search query with no additional context or explanations.
'''
) + formatting

search_query_prompt_single_ex = (
    "You are tasked with answering a complex, multihop question using a similarity search engine for Wikipedia. "
    "To tackle this, you need to break down the question into simpler parts. Begin by generating a search query "
    "to find information about one of the key components or entities involved in the question. IMPORTANT: avoid repeating search queries that already appear in the context or searching for documents that already are in the context."
    "IMPORTANT: Some context documents are wrong so do not keep searching for them. Make sure your query "
    "is focused just on identifying or understanding this singular part or entity, without delving into how it connects "
    "to the overall question.  "
    '''Here are some examples of successful search queries and their desired functionality:
Question: What is Jacob Kraemer's occupation?
Query: Jacob Kraemer
Retrieved Document: Title: Jacob Kraemer Content: Jacob Kraemer is a Canadian actor, from Fonthill, Ontario. He became known to young audiences after his role in "The Elizabeth Smart Story" and as Ben on Disney and Family's "Naturally, Sadie".
Answer: actor

Question: What genre is Drôles de zèbres?
Query: Drôles de zèbres
Retrieved Document: Title: Drôles de zèbres Content: Drôles de zèbres (English: Funny Zebras ) is a 1977 French comedy film.

Please provide only the search query with no additional context or explanations.
'''
) + formatting

triviaqa_search_query_prompt = (
    "You are tasked with answering a factual question using a similarity search engine for Wikipedia."
    "To tackle this, focus on generating a precise search query to find information about the key entity or concept directly related to the question. "
    "IMPORTANT: avoid repeating search queries that already appear in the context or searching for documents that already are in the context. "
    "Make sure your query is concise and specific to the exact entity or concept you need to understand, without exploring additional connections. "
    "Please provide only the search query with no additional context or explanations."
) + formatting



abstain_force_answer_prompt = (
    "You are a knowledgeable question-answering assistant. Answer the following "
    "question, and provide a brief explanation of your reasoning. If you are unsure, "
    "respond with 'unsure'."
) + formatting_abstain

