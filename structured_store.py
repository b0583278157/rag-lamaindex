import json

def load_structured_data(path="knowledge_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)