from extractor import load_documents

docs = load_documents()

print(len(docs))
print(docs[0].text[:500])