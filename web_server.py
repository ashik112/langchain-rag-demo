from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import json
from rag_system import RAGSystem
import os
import time

app = Flask(__name__, static_folder='web')
CORS(app)

# Initialize RAG system
rag = RAGSystem()

# Try to load existing vector store
if not rag.load_vector_store():
    # If no vector store exists, process documents
    documents = rag.load_documents()
    chunks = rag.process_documents(documents)
    rag.create_vector_store(chunks)

# Set up the QA chain
rag.setup_qa_chain()

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

@app.route('/api/rag', methods=['POST'])
def query_rag():
    try:
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        print(f"Received question: {question}")

        # Get the full response with source documents
        response = rag.qa_chain.invoke({"question": question})
        answer = response["answer"]
        source_documents = response.get("source_documents", [])
        
        # Process source documents for better information
        sources_info = []
        for i, doc in enumerate(source_documents):
            source_info = {
                'source': doc.metadata.get("source", "Unknown"),
                'chunk_id': doc.metadata.get("chunk_id", i),
                'summary': doc.metadata.get("summary", ""),
                'relevance_score': getattr(doc, 'score', None),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources_info.append(source_info)
        
        # Create a more detailed sources section
        if sources_info:
            sources_text = "\n\n## Sources\n\n"
            unique_sources = {}
            for source in sources_info:
                source_name = source['source']
                if source_name not in unique_sources:
                    unique_sources[source_name] = []
                unique_sources[source_name].append(source)
            
            for source_name, chunks in unique_sources.items():
                sources_text += f"**{source_name}**\n"
                for chunk in chunks:
                    if chunk['summary']:
                        sources_text += f"- {chunk['summary']}\n"
                sources_text += "\n"
            
            answer += sources_text
        
        print(f"Sending response with {len(source_documents)} source documents")
        
        return jsonify({
            'answer': answer,
            'sources': [s['source'] for s in sources_info],
            'source_details': sources_info,
            'total_sources': len(source_documents)
        })
        
    except Exception as e:
        print(f"Error in query_rag: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True) 