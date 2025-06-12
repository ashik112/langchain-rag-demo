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

        def generate():
            try:
                # Stream the response token by token
                for chunk in rag.stream_query(question):
                    if chunk:
                        print(f"Sending chunk: {chunk}")
                        # Send each chunk as a separate event
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                        # Force flush
                        time.sleep(0.01)  # Small delay to ensure chunks are sent separately
            except Exception as e:
                print(f"Error in generate: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        response = Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Transfer-Encoding': 'chunked'
            }
        )
        return response
    except Exception as e:
        print(f"Error in query_rag: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True) 