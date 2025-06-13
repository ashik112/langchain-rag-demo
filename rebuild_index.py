#!/usr/bin/env python3

import os
from rag_system import RAGSystem

def main():
    print("🔄 Rebuilding vector store with all documents...")
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Force document loading
    print("📚 Loading documents...")
    documents = rag.load_documents()
    print(f"📄 Loaded {len(documents)} documents")
    
    if len(documents) == 0:
        print("❌ No documents loaded! Check the assets directory.")
        return
    
    # Process documents
    print("⚙️  Processing documents...")
    chunks = rag.process_documents(documents)
    print(f"📦 Created {len(chunks)} chunks")
    
    # Create vector store
    print("🗃️  Creating vector store...")
    rag.create_vector_store(chunks)
    print("✅ Vector store created successfully!")
    
    print("🎉 Rebuild complete! The new Goama playbook should now be included.")

if __name__ == "__main__":
    main() 