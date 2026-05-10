import os
from typing import Any

from pymongo import MongoClient

from research import research_app, ResearcherState


def run_test() -> None:
    initial_state: ResearcherState = {
        "query": "transformers",
        "agent_id": "test_agent_01",
        "search_count": 0,
        "approved_papers": [],
        "raw_papers": []
    }

    print(f"Starting test for: '{initial_state['query']}'")
    print("-" * 50)

    for step in research_app.stream(initial_state):
        for node_name, state_update in step.items():
            print(f"\n[ Node: {node_name} ]")

            if "raw_papers" in state_update:
                found = len(state_update["raw_papers"])
                print(f" -> Found {found} papers.")

            if "approved_papers" in state_update:
                approved = len(state_update["approved_papers"])
                print(f" -> Total approved: {approved}")

            if "search_count" in state_update:
                count = state_update["search_count"]
                print(f" -> Search loop count: {count}")

    print("\n" + "=" * 50)
    print("Graph Execution Complete.")

    mongo_uri = os.getenv(
        "MONGO_URI", 
        "mongodb://localhost:27017"
    )
    client: MongoClient = MongoClient(mongo_uri)
    db = client["agent_test_agent_01"]
    collection = db["approved_papers"]
    
    saved_count = collection.count_documents({})
    print(f"Documents saved to MongoDB: {saved_count}")

    if saved_count > 0:
        print("\n[ Inspecting First Saved Document ]")
        sample_doc = collection.find_one()
        if sample_doc:
            title = sample_doc.get("title", "No Title")
            print(f"Title: {title}")
            
            abstract = sample_doc.get("abstract", "")
            print(f"Abstract: {abstract[:100]}...")
            
            vector = sample_doc.get("abstract_embedding", [])
            print(f"Vector Length: {len(vector)}")
            print(f"Vector Snippet: {vector[:3]}...")


if __name__ == "__main__":
    run_test()
