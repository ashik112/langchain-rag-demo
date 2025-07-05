from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import json
from rag_system import RAGSystem
import os
import time
import uuid

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

# Set up shared components for sessions
rag.setup_shared_components()

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

@app.route('/api/create-session', methods=['POST'])
def create_session():
    """Create a new session for a page load."""
    try:
        session_id = str(uuid.uuid4())
        rag.create_session(session_id)
        
        print(f"🆕 Created new page session: {session_id}")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'New session created'
        })
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/destroy-session', methods=['POST'])
def destroy_session():
    """Destroy a session when page closes."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400
        
        destroyed = rag.destroy_session(session_id)
        
        if destroyed:
            print(f"🗑️ Destroyed page session: {session_id}")
            return jsonify({'success': True, 'message': 'Session destroyed'})
        else:
            return jsonify({'success': False, 'message': 'Session not found'})
    except Exception as e:
        print(f"❌ Error destroying session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-session', methods=['POST'])
def verify_session():
    """Verify if a session exists and is valid."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400
        
        exists = rag.session_exists(session_id)
        
        if exists:
            print(f"✅ Session {session_id} verified")
            return jsonify({'success': True, 'message': 'Session is valid'})
        else:
            print(f"❌ Session {session_id} not found")
            return jsonify({'error': 'Session not found or expired'}), 404
            
    except Exception as e:
        print(f"❌ Error verifying session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions-info', methods=['GET'])
def sessions_info():
    """Get information about active sessions."""
    try:
        info = {
            'session_count': rag.get_session_count(),
            'sessions': rag.get_session_info()
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup-sessions', methods=['POST'])
def cleanup_sessions():
    """Clean up old sessions."""
    try:
        data = request.get_json() or {}
        max_age = data.get('max_age_seconds', 3600)  # Default 1 hour
        
        cleaned = rag.cleanup_old_sessions(max_age)
        return jsonify({
            'success': True,
            'cleaned_sessions': cleaned,
            'remaining_sessions': rag.get_session_count()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag', methods=['POST'])
def query_rag():
    """Main API endpoint for processing RAG queries with hybrid document + general knowledge approach."""
    try:
        # Extract and validate the user's question and session
        data = request.get_json()
        question = data.get('question')
        session_id = data.get('session_id')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400
            
        if not rag.session_exists(session_id):
            return jsonify({'error': 'Session not found or expired'}), 400

        print(f"🔍 Processing question for session {session_id}: {question}")

        # Use session-specific query method
        response = rag.query_with_session(question, session_id)
        answer = response["answer"]
        source_documents = response.get("source_documents", [])
        
        print(f"📚 Found {len(source_documents)} relevant document chunks")
        
        # Process and analyze source documents for transparency
        sources_info = []
        document_topics = set()
        
        for i, doc in enumerate(source_documents):
            source_info = {
                'source': doc.metadata.get("source", "Unknown"),
                'chunk_id': doc.metadata.get("chunk_id", i),
                'summary': doc.metadata.get("summary", ""),
                'relevance_score': getattr(doc, 'score', None),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources_info.append(source_info)
            
            content_lower = doc.page_content.lower()
            tech_keywords = ['api', 'integration', 'android', 'ios', 'payment', 'authentication', 
                           'database', 'sdk', 'javascript', 'python', 'java', 'swift', 'kotlin']
            for keyword in tech_keywords:
                if keyword in content_lower:
                    document_topics.add(keyword)
        
        # Calculate token usage for cost tracking
        input_text = question
        if source_documents:
            context_text = "\n".join([doc.page_content for doc in source_documents])
            input_text += f"\n{context_text}"
        
        estimated_input_tokens = len(input_text) // 4
        estimated_output_tokens = len(answer) // 4
        total_tokens = estimated_input_tokens + estimated_output_tokens
        
        estimated_input_cost = (estimated_input_tokens / 1000) * 0.00015
        estimated_output_cost = (estimated_output_tokens / 1000) * 0.0006
        total_estimated_cost = estimated_input_cost + estimated_output_cost
        
        token_usage = {
            'input_tokens': estimated_input_tokens,
            'output_tokens': estimated_output_tokens,
            'total_tokens': total_tokens,
            'context_docs_count': len(source_documents),
            'estimated_cost': {
                'input_cost_usd': round(estimated_input_cost, 6),
                'output_cost_usd': round(estimated_output_cost, 6),
                'total_cost_usd': round(total_estimated_cost, 6)
            }
        }
        
        return jsonify({
            'answer': answer,
            'session_id': session_id,
            'sources': [s['source'] for s in sources_info],
            'source_details': sources_info,
            'total_sources': len(source_documents),
            'document_topics': list(document_topics),
            'response_type': 'hybrid',
            'has_code_examples': '```' in answer,
            'token_usage': token_usage,
            'timestamp': time.time()
        })
        
    except Exception as e:
        app.logger.error(f"Error processing RAG request: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'details': {
                'type': type(e).__name__,
                'message': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 500

if __name__ == '__main__':
    app.logger.info("Starting Flask server in debug mode")
    app.run(
        host="0.0.0.0", 
        port=5000, 
        debug=True, 
        threaded=True,
        use_reloader=True,
        extra_files=['rag_system.py']
    ) 