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
    """
    Main API endpoint for processing RAG queries with hybrid document + general knowledge approach.
    
    This endpoint:
    1. Receives user questions
    2. Retrieves relevant document context
    3. Generates hybrid responses (documents + relevant general knowledge)
    4. Returns structured response with source attribution
    """
    try:
        # Extract and validate the user's question
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        print(f"🔍 Processing question: {question}")

        # Step 1: Get comprehensive response from the hybrid RAG chain
        # This includes both document-based info and relevant general knowledge
        response = rag.qa_chain.invoke({"question": question})
        answer = response["answer"]
        source_documents = response.get("source_documents", [])
        
        print(f"📚 Found {len(source_documents)} relevant document chunks")
        
        # Step 2: Process and analyze source documents for transparency
        # This helps users understand what information came from their documents
        sources_info = []
        document_topics = set()  # Track topics mentioned in documents
        
        for i, doc in enumerate(source_documents):
            # Extract metadata and content preview for each source
            source_info = {
                'source': doc.metadata.get("source", "Unknown"),
                'chunk_id': doc.metadata.get("chunk_id", i),
                'summary': doc.metadata.get("summary", ""),
                'relevance_score': getattr(doc, 'score', None),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources_info.append(source_info)
            
            # Extract key topics/concepts from document content for context awareness
            # This helps determine if the response should include general knowledge
            content_lower = doc.page_content.lower()
            
            # Identify technical topics mentioned in documents
            tech_keywords = ['api', 'integration', 'android', 'ios', 'payment', 'authentication', 
                           'database', 'sdk', 'javascript', 'python', 'java', 'swift', 'kotlin']
            for keyword in tech_keywords:
                if keyword in content_lower:
                    document_topics.add(keyword)
        
        print(f"🏷️  Document topics identified: {document_topics}")
        
        # Step 3: Keep the answer clean without source references
        # The AI will respond naturally without mentioning documents
        # (Sources are still tracked internally for debugging/analytics)
        
        # Step 4: Calculate token usage for cost tracking and optimization
        # This helps users understand the computational cost of their queries
        input_text = question
        if source_documents:
            # Include context in token calculation
            context_text = "\n".join([doc.page_content for doc in source_documents])
            input_text += f"\n{context_text}"
        
        # Estimate token counts (1 token ≈ 4 characters for most models)
        # Note: This is an approximation - actual tokenization may vary
        estimated_input_tokens = len(input_text) // 4
        estimated_output_tokens = len(answer) // 4
        total_tokens = estimated_input_tokens + estimated_output_tokens
        
        # Calculate estimated costs based on Google Gemini pricing
        # Input: ~$0.00015 per 1K tokens, Output: ~$0.0006 per 1K tokens
        estimated_input_cost = (estimated_input_tokens / 1000) * 0.00015
        estimated_output_cost = (estimated_output_tokens / 1000) * 0.0006
        total_estimated_cost = estimated_input_cost + estimated_output_cost
        
        # Create comprehensive token usage information
        token_usage = {
            'input_tokens': estimated_input_tokens,
            'output_tokens': estimated_output_tokens,
            'total_tokens': total_tokens,
            'context_docs_count': len(source_documents),
            'context_length': sum(len(doc.page_content) for doc in source_documents),
            'question_length': len(question),
            'answer_length': len(answer),
            'estimated_cost': {
                'input_cost_usd': round(estimated_input_cost, 6),
                'output_cost_usd': round(estimated_output_cost, 6),
                'total_cost_usd': round(total_estimated_cost, 6)
            },
            'efficiency_metrics': {
                'tokens_per_context_doc': round(estimated_input_tokens / max(len(source_documents), 1), 2),
                'output_input_ratio': round(estimated_output_tokens / max(estimated_input_tokens, 1), 2),
                'cost_per_response_cents': round(total_estimated_cost * 100, 4)
            }
        }
        
        print(f"💰 Token usage: {total_tokens} total ({estimated_input_tokens} in, {estimated_output_tokens} out)")
        print(f"💰 Estimated cost: ${total_estimated_cost:.6f} USD")
        
        # Step 5: Determine response type for UI indication
        # This helps users understand if the response is purely document-based or hybrid
        response_type = "hybrid"  # Default assumption
        
        # Simple heuristics to detect if response includes general knowledge
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in [
            "implementation details", "code example", "here's how", "you can implement",
            "general approach", "common pattern", "typical implementation"
        ]):
            response_type = "hybrid"
        elif len(source_documents) > 0 and not any(phrase in answer_lower for phrase in [
            "however", "additionally", "to implement this", "here's an example"
        ]):
            response_type = "document_only"
        else:
            response_type = "hybrid"
        
        print(f"📊 Response type: {response_type}")
        print(f"✅ Sending response with {len(source_documents)} source documents")
        
        # Step 6: Return structured response with all metadata including token usage
        return jsonify({
            'answer': answer,
            'sources': [s['source'] for s in sources_info],
            'source_details': sources_info,
            'total_sources': len(source_documents),
            'document_topics': list(document_topics),
            'response_type': response_type,  # Indicates if hybrid or document-only
            'has_code_examples': '```' in answer,  # Indicates if response includes code
            'token_usage': token_usage,  # Comprehensive token tracking and cost estimation
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"❌ Error in query_rag: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True) 