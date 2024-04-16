import os
import re
import hashlib
import ast
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations

def calculate_similarity(text1, text2, algorithm='sequence_matcher'):
    """
    Calculate the similarity percentage between two texts using different algorithms.
    """
    if algorithm == 'sequence_matcher':
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio() * 100
    elif algorithm == 'document_fingerprint':
        fingerprint1 = hash_document(text1)
        fingerprint2 = hash_document(text2)
        intersection = len(fingerprint1 & fingerprint2)
        union = len(fingerprint1 | fingerprint2)
        return (intersection / union) * 100
    elif algorithm == 'token_based_similarity':
        tokens1 = tokenize_code(text1)
        tokens2 = tokenize_code(text2)
        return calculate_token_similarity(tokens1, tokens2)
    elif algorithm == 'ast_comparison':
        ast1 = generate_ast(text1)
        ast2 = generate_ast(text2)
        return calculate_ast_similarity(ast1, ast2)
    elif algorithm == 'function_mapping':
        return calculate_function_similarity(text1, text2)
    elif algorithm == 'clone_detection':
        return calculate_clone_similarity(text1, text2)
    elif algorithm == 'shingling':
        shingles1 = generate_shingles(text1)
        shingles2 = generate_shingles(text2)
        return calculate_shingle_similarity(shingles1, shingles2)
    elif algorithm == 'semantic_analysis':
        return calculate_semantic_similarity(text1, text2)

def hash_document(text):
    """
    Generate a document fingerprint using hashing.
    """
    words = re.findall(r'\w+', text.lower())
    return set(hashlib.sha1(word.encode()).hexdigest() for word in words)

def tokenize_code(code):
    """
    Tokenize code into keywords, identifiers, literals, etc.
    """
    # Sample implementation, actual implementation may vary based on programming language
    return re.findall(r'\b\w+\b', code)

def calculate_token_similarity(tokens1, tokens2):
    """
    Calculate similarity based on token sequences.
    """
    common_tokens = len(set(tokens1) & set(tokens2))
    total_tokens = len(set(tokens1) | set(tokens2))
    return (common_tokens / total_tokens) * 100

def generate_ast(code):
    """
    Generate Abstract Syntax Tree (AST) from code.
    """
    try:
        return ast.parse(code)
    except SyntaxError:
        return None

def calculate_ast_similarity(ast1, ast2):
    """
    Compare ASTs to detect similarities.
    """
    if ast1 is None or ast2 is None:
        return 0.0
    return SequenceMatcher(None, ast.dump(ast1), ast.dump(ast2)).ratio() * 100

def calculate_function_similarity(code1, code2):
    """
    Identify similar functions or methods across codebases.
    """
    # Sample implementation, actual implementation may vary based on programming language
    function_names1 = set(re.findall(r'def\s+(\w+)\s*\(', code1))
    function_names2 = set(re.findall(r'def\s+(\w+)\s*\(', code2))
    common_functions = len(function_names1 & function_names2)
    total_functions = len(function_names1 | function_names2)

    if total_functions == 0:
        return 0.0
    else:
        return (common_functions / total_functions) * 100

def calculate_clone_similarity(text1, text2):
    """
    Detect code clones using token-based hashing.
    """
    # Sample implementation, actual implementation may vary based on programming language
    hashset1 = set(hash(token) for token in tokenize_code(text1))
    hashset2 = set(hash(token) for token in tokenize_code(text2))
    intersection = len(hashset1 & hashset2)
    union = len(hashset1 | hashset2)
    return (intersection / union) * 100

def generate_shingles(text, k=5):
    """
    Generate shingles from text.
    """
    shingles = set()
    words = text.split()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i + k])
        shingles.add(hash(shingle))
    return shingles

def calculate_shingle_similarity(shingles1, shingles2):
    """
    Calculate similarity based on shingles.
    """
    common_shingles = len(shingles1 & shingles2)
    total_shingles = len(shingles1 | shingles2)
    return (common_shingles / total_shingles) * 100

def calculate_semantic_similarity(text1, text2):
    """
    Analyze semantic meaning of code to detect plagiarism.
    """
    # Sample implementation, actual implementation may vary based on programming language
    # Here, we'll use token-based similarity as a proxy for semantic analysis
    return calculate_token_similarity(tokenize_code(text1), tokenize_code(text2))

def read_text_file(file_path):
    """
    Read and return the content of a text file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def check_plagiarism(folder_path, threshold):
    """
    Check plagiarism among text files in the given folder using different algorithms.
    """
    # Dictionary to store plagiarism results
    plagiarism_results = defaultdict(list)

    # List all text files in the folder
    text_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    # Iterate through each pair of text files
    for file1, file2 in combinations(text_files, 2):
        text1 = read_text_file(os.path.join(folder_path, file1))
        text2 = read_text_file(os.path.join(folder_path, file2))

        similarity_results = {
            'sequence_matcher': calculate_similarity(text1, text2, 'sequence_matcher'),
            'document_fingerprint': calculate_similarity(text1, text2, 'document_fingerprint'),
            'token_based_similarity': calculate_similarity(text1, text2, 'token_based_similarity'),
            'ast_comparison': calculate_similarity(text1, text2, 'ast_comparison'),
            'function_mapping': calculate_similarity(text1, text2, 'function_mapping'),
            'clone_detection': calculate_similarity(text1, text2, 'clone_detection'),
            'shingling': calculate_similarity(text1, text2, 'shingling'),
            'semantic_analysis': calculate_similarity(text1, text2, 'semantic_analysis')
        }

        max_algorithm = max(similarity_results, key=similarity_results.get)
        max_similarity = similarity_results[max_algorithm]

        if max_similarity >= threshold:
            plagiarism_results[file1].append((file2, max_similarity, max_algorithm))
            plagiarism_results[file2].append((file1, max_similarity, max_algorithm))

    # Print plagiarism results
    for file, similar_files in plagiarism_results.items():
        if similar_files:
            print(f"Plagiarism detected in '{file}':")
            for similar_file, similarity, algorithm in similar_files:
                print(f"- '{similar_file}': {similarity:.2f}% similar (Algorithm: {algorithm})")

# Example usage
if __name__ == "__main__":
    folder_path = input("Enter the folder path containing text files: ")
    threshold = float(input("Enter the plagiarism threshold percentage (0-100): "))
    check_plagiarism(folder_path, threshold)
