import csv
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import difflib
import pandas as pd

@dataclass
class Result:
    score: float
    response: str
    metadata: Dict[str, Any] = None

def load_results(file_path: str) -> List[Result]:
    """Load results from a CSV file."""
    results = []
    df = pd.read_csv(file_path)
    
    for _, row in df.iterrows():
        result = Result(
            score=float(row.get('score', 0.0)),
            response=str(row.get('response', '')),
            metadata={k: v for k, v in row.items() if k not in ['score', 'response']}
        )
        results.append(result)
    
    return results

def compare_results(result1: Result, result2: Result) -> Dict[str, Any]:
    """
    Compare two results and return detailed analysis.
    """
    analysis = {
        'score_match': abs(result1.score - result2.score) < 0.0001,  # Using small epsilon for float comparison
        'score_diff': abs(result1.score - result2.score),
        'response_match': result1.response == result2.response,
        'response_similarity': difflib.SequenceMatcher(None, result1.response, result2.response).ratio(),
        'metadata_match': result1.metadata == result2.metadata if result1.metadata and result2.metadata else None
    }
    
    return analysis

def analyze_results(file1: str, file2: str) -> Dict[str, Any]:
    """
    Analyze two result files and return comparison results.
    """
    results1 = load_results(file1)
    results2 = load_results(file2)
    
    if len(results1) != len(results2):
        print(f"Warning: Files have different number of results. File1: {len(results1)}, File2: {len(results2)}")
    
    comparisons = []
    for r1, r2 in zip(results1, results2):
        comparison = compare_results(r1, r2)
        comparisons.append(comparison)
    
    # Aggregate results
    aggregated = {
        'total_comparisons': len(comparisons),
        'perfect_matches': sum(1 for c in comparisons if c['score_match'] and c['response_match']),
        'score_matches_only': sum(1 for c in comparisons if c['score_match'] and not c['response_match']),
        'response_matches_only': sum(1 for c in comparisons if not c['score_match'] and c['response_match']),
        'no_matches': sum(1 for c in comparisons if not c['score_match'] and not c['response_match']),
        'avg_score_diff': sum(c['score_diff'] for c in comparisons) / len(comparisons),
        'avg_response_similarity': sum(c['response_similarity'] for c in comparisons) / len(comparisons)
    }
    
    return aggregated

def print_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print the analysis results in a readable format.
    """
    print("\n=== Analysis Results ===")
    print(f"Total Comparisons: {analysis['total_comparisons']}")
    print(f"Perfect Matches: {analysis['perfect_matches']} ({analysis['perfect_matches']/analysis['total_comparisons']:.2%})")
    print(f"Score Matches Only: {analysis['score_matches_only']} ({analysis['score_matches_only']/analysis['total_comparisons']:.2%})")
    print(f"Response Matches Only: {analysis['response_matches_only']} ({analysis['response_matches_only']/analysis['total_comparisons']:.2%})")
    print(f"No Matches: {analysis['no_matches']} ({analysis['no_matches']/analysis['total_comparisons']:.2%})")
    print(f"\nAverage Score Difference: {analysis['avg_score_diff']:.6f}")
    print(f"Average Response Similarity: {analysis['avg_response_similarity']:.2%}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python analysis.py <results1.csv> <results2.csv>")
        sys.exit(1)
    
    result1_path = sys.argv[1]
    result2_path = sys.argv[2]
    
    analysis = analyze_results(result1_path, result2_path)
    print_analysis(analysis) 