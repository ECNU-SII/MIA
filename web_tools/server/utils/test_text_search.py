import sys
import os
import json

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set environment variables for Jina scraper and LLM

from utils.text_search import call_web_text_search

def test_call_web_text_search():
    all_results = {}
    
    # Test with a simple query
    query = "AI development trends"
    result = call_web_text_search(query, max_results=3)
    all_results["Test 1 - Simple query"] = result
    print("Test 1 - Simple query:")
    print(f"Success: {result['success']}")
    print(f"Engine: {result['engine']}")
    print(f"Number of results: {len(result['results'])}")
    print(f"Latency: {result['latency_ms']} ms")
    print()



    # Test with a query and no info_to_extract (should default to query)

    
    # Save results to JSON file
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("Results saved to test_results.json")

if __name__ == "__main__":
    test_call_web_text_search()