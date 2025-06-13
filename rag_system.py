import os
import re
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import time

os.environ["GOOGLE_API_KEY"] = "AIzaSyAzbenL0ic4sltL1kENQPYVd8l_zNgPy1I"; 

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, assets_dir: str = "assets"):
        """Initialize the RAG system.
        
        Args:
            assets_dir (str): Directory containing the documents to be processed
        """
        self.assets_dir = assets_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.qa_chain = None
    
    def clean_text(self, text: str) -> str:
        """Minimal text cleaning - just remove invisible Unicode characters."""
        if not text:
            return text
        
        # Only remove zero-width spaces and other invisible Unicode characters
        # Don't modify any visible text or spacing
        text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', '', text)
        
        return text

    def load_documents(self) -> List:
        """Load documents from the assets directory, dispatching by file extension."""
        print(f"🔍 Starting document loading from '{self.assets_dir}'...")
        
        # Use selective loaders based on platform and file type
        import platform
        is_windows = platform.system().lower() == "windows"
        
        if is_windows:
            # On Windows, use more reliable loaders to avoid hanging
            loaders = {
                ".txt": (TextLoader, {"encoding": "utf-8"}),
                ".pdf": (PyPDFLoader, {}),  # Use PyPDFLoader for PDFs on Windows (more reliable)
                ".docx": (Docx2txtLoader, {}),  # Use traditional loader for DOCX on Windows
            }
            print("🪟 Windows detected - using Windows-optimized loaders")
        else:
            # On Unix systems, use unstructured loaders
            loaders = {
                ".txt": (TextLoader, {"encoding": "utf-8"}),
                ".pdf": (UnstructuredPDFLoader, {"mode": "single", "strategy": "fast"}),
                ".docx": (UnstructuredWordDocumentLoader, {"mode": "single"}),
            }
            print("🐧 Unix system detected - using unstructured loaders")

        documents = []
        total_files_found = 0
        
        for root, _, files in os.walk(self.assets_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in loaders:
                    total_files_found += 1
                    
        print(f"📁 Found {total_files_found} supported files to process")
        
        for root, _, files in os.walk(self.assets_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                entry = loaders.get(ext)
                if not entry:
                    print(f"⏭️  Skipping unsupported file type: {fn}")
                    continue  # skip unsupported types

                loader_cls, loader_kwargs = entry
                path = os.path.join(root, fn)
                print(f"📄 Processing {ext.upper()} file: {fn}")
                
                success = False
                
                # Try the primary loader first
                try:
                    loader_name = "Unstructured" if "Unstructured" in loader_cls.__name__ else "Standard"
                    print(f"🔄 Loading {ext.upper()} with {loader_name} loader: '{fn}'...")
                    
                    loader = loader_cls(path, **loader_kwargs)
                    docs = loader.load()
                    
                    if docs and any(doc.page_content.strip() for doc in docs):
                        # Clean text content for all documents
                        for doc in docs:
                            if doc.page_content:
                                original_content = doc.page_content
                                cleaned_content = self.clean_text(original_content)
                                doc.page_content = cleaned_content
                                
                                # Log cleaning results for debugging
                                if len(original_content) != len(cleaned_content):
                                    print(f"🧹 Cleaned {ext.upper()} text: {len(original_content)} -> {len(cleaned_content)} chars")
                        
                        documents.extend(docs)
                        print(f"✅ Successfully loaded '{fn}' with {loader_name} loader ({len(docs)} chunks)")
                        success = True
                    else:
                        print(f"⚠️  {loader_name} loader returned empty content for '{fn}'")
                except Exception as e:
                    print(f"❌ Primary loader failed for '{fn}': {e}")
                
                # Fallback to traditional loaders if unstructured fails
                if not success:
                    fallback_loader = None
                    fallback_kwargs = {}
                    
                    if ext == ".pdf":
                        fallback_loader = PyPDFLoader
                        fallback_kwargs = {}
                        fallback_name = "PyPDFLoader"
                    elif ext == ".docx":
                        fallback_loader = Docx2txtLoader
                        fallback_kwargs = {}
                        fallback_name = "Docx2txtLoader"
                    else:
                        fallback_loader = loader_cls
                        fallback_kwargs = loader_kwargs
                        fallback_name = "Standard loader"
                    
                    if fallback_loader:
                        try:
                            print(f"🔄 Trying fallback {fallback_name} for '{fn}'...")
                            fallback = fallback_loader(path, **fallback_kwargs)
                            docs = fallback.load()
                            
                            if docs and any(doc.page_content.strip() for doc in docs):
                                # Clean the fallback content too
                                for doc in docs:
                                    if doc.page_content:
                                        original_content = doc.page_content
                                        cleaned_content = self.clean_text(original_content)
                                        doc.page_content = cleaned_content
                                        
                                        if len(original_content) != len(cleaned_content):
                                            print(f"🧹 Cleaned {ext.upper()} text: {len(original_content)} -> {len(cleaned_content)} chars")
                                
                                documents.extend(docs)
                                print(f"✅ Successfully loaded '{fn}' with {fallback_name} ({len(docs)} chunks)")
                                success = True
                            else:
                                print(f"⚠️  {fallback_name} returned empty content for '{fn}'")
                        except Exception as fallback_error:
                            print(f"❌ {fallback_name} also failed for '{fn}': {fallback_error}")
                
                if not success:
                    print(f"💥 Failed to load '{fn}' with any available loader")

        print(f"📚 Document loading complete: {len(documents)} total document chunks loaded")
        
        if len(documents) == 0:
            print("⚠️  WARNING: No documents were successfully loaded!")
            print("🔍 Please check:")
            print("   - File permissions in the assets directory")
            print("   - File formats are supported (.txt, .pdf, .docx)")
            print("   - Files are not corrupted")
        
        return documents

        
    # def load_documents(self) -> List:
    #     """Load documents from the assets directory."""
    #     # Configure loaders for different file types
    #     loaders = {
    #         ".txt": TextLoader,
    #         ".pdf": UnstructuredPDFLoader,
    #         ".docx": Docx2txtLoader,
    #     }
        
    #     # Create a directory loader with the configured loaders
    #     loader = DirectoryLoader(
    #         self.assets_dir,
    #         glob="**/*",
    #         loader_mapping={  # note: loader_mapping, *not* loader_cls
    #             ".txt": TextLoader,
    #             ".pdf": UnstructuredPDFLoader,
    #             ".docx": Docx2txtLoader,
    #         }
    #     )
    #     documents = loader.load()
    #     print(f"Loaded {len(documents)} documents")
    #     return documents
    
    def process_documents(self, documents: List) -> List:
        """Process and chunk the documents with improved strategy."""
        # Use a more sophisticated text splitter with better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased for better context
            chunk_overlap=300,  # Increased overlap for better continuity
            length_function=len,
            separators=[
                "\n\n\n",  # Triple newlines (major sections)
                "\n\n",    # Double newlines (paragraphs)
                "\n",      # Single newlines
                ". ",      # Sentences
                "! ",      # Exclamations
                "? ",      # Questions
                "; ",      # Semicolons
                ", ",      # Commas
                " ",       # Spaces
                ""         # Characters
            ],
            keep_separator=True,  # Keep separators for better context
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to chunks for better retrieval
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
            # Add first few words as a summary
            words = chunk.page_content.split()[:10]
            chunk.metadata['summary'] = ' '.join(words) + '...'
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_vector_store(self, chunks: List):
        """Create and save the FAISS vector store."""
        # Create the vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the vector store locally
        self.vector_store.save_local("faiss_index")
        print("Vector store created and saved locally")
    
    def load_vector_store(self) -> bool:
        """Load the vector store from disk. Returns True if loaded, False otherwise."""
        store_path = "faiss_index"
        # Debug: what files do we actually see?
        print("🔍 Checking for existing index in", os.getcwd(), "…")
        print("🔍 Contents of cwd:", os.listdir("."))

        if os.path.exists(store_path):
            self.vector_store = FAISS.load_local(
                store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded vector store from '{store_path}'")
            return True

        print(f"⚠️  No vector store found at '{store_path}'")
        return False
    
    def setup_qa_chain(self):
        """Set up the question-answering chain with hybrid document + general knowledge approach."""
        # Initialize the language model with hybrid system prompt
        # This allows the AI to provide relevant general knowledge when documents are incomplete
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0.3,  # Lower temperature for more focused responses
            disable_streaming=False,  # Disable streaming
            model_kwargs={
                "system_instruction": """You are a knowledgeable AI assistant specializing in gaming platforms, tournament systems, and technical integrations. You provide helpful, conversational responses as if you naturally know this information.

RESPONSE STYLE:
- Act like a friendly, knowledgeable assistant
- Never mention "documents", "sources", or "based on the information provided"
- Speak naturally as if you inherently know this information
- Be conversational and helpful
- Provide specific details and examples when relevant

KNOWLEDGE SCOPE:
- Gaming platform integrations
- Tournament systems and APIs
- Payment processing for games
- SDK implementations
- Mobile game development (Android/iOS)
- Web-based game integrations
- Technical implementation details

FORMATTING GUIDELINES:
- Use clear markdown headings (# ## ###) when organizing information
- Use bullet points (-) for features and lists
- Use numbered lists (1. 2. 3.) for step-by-step processes
- Use **bold** for important terms and concepts
- Use `code formatting` for technical terms, API endpoints, and parameters
- Use ```language blocks for code examples
- Keep responses well-structured and easy to read

RESPONSE APPROACH:
- Answer directly and confidently
- Provide practical implementation guidance
- Include relevant code examples when helpful
- Explain technical concepts clearly
- Offer additional context that would be useful

Remember: Be helpful, knowledgeable, and conversational. Never reference documents or sources - just provide the information naturally."""
            }
        )
        
        # Set up conversation memory to maintain context across questions
        # This helps the AI understand the ongoing conversation and document scope
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer" 
        )
        
        # Create a sophisticated retriever that finds diverse, relevant content
        # MMR (Maximum Marginal Relevance) balances relevance with diversity
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 6,  # Get more chunks initially for better context
                "fetch_k": 12,  # Fetch more candidates for MMR selection
                "lambda_mult": 0.7  # Balance between relevance (1.0) and diversity (0.0)
            }
        )
        
        # Create custom prompt template for hybrid responses
        # This template guides the AI to properly combine document and general knowledge
        from langchain.prompts import PromptTemplate
        
        custom_prompt = PromptTemplate(
            template="""You are a knowledgeable AI assistant. Use the following context and conversation history to provide a helpful, natural response.

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide a natural, conversational response
- Never mention "documents", "sources", or "based on the information provided"
- Act as if you naturally know this information
- Be helpful and provide specific details when relevant
- Use proper markdown formatting for readability
- If the question is outside your knowledge scope, politely explain what you can help with instead

RESPONSE:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create the conversational QA chain with hybrid capabilities
        # This chain combines document retrieval with conversational memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=base_retriever,
            memory=memory,
            return_source_documents=True,  # Include source docs for transparency
            output_key="answer",
            return_generated_question=True,  # Help with follow-up questions
            combine_docs_chain_kwargs={"prompt": custom_prompt}  # Use our hybrid prompt
        )
    
    def analyze_query_intent(self, question: str) -> dict:
        """
        Analyze the user's query to determine intent and suggest response strategy.
        
        This helps the hybrid system understand:
        1. What type of information the user is seeking
        2. Whether they need implementation details vs. conceptual info
        3. If the query relates to topics likely in technical documents
        
        Args:
            question (str): The user's question
            
        Returns:
            dict: Analysis results with intent classification and keywords
        """
        question_lower = question.lower()
        
        # Intent classification based on question patterns
        intent_patterns = {
            'implementation': ['how to', 'how do i', 'implement', 'integrate', 'code', 'example', 'tutorial'],
            'explanation': ['what is', 'what are', 'explain', 'describe', 'definition', 'meaning'],
            'comparison': ['difference', 'compare', 'vs', 'versus', 'better', 'best'],
            'troubleshooting': ['error', 'problem', 'issue', 'fix', 'debug', 'not working'],
            'features': ['features', 'capabilities', 'can i', 'does it', 'support'],
            'getting_started': ['start', 'begin', 'setup', 'install', 'first', 'initial']
        }
        
        # Detect primary intent
        detected_intents = []
        for intent, patterns in intent_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                detected_intents.append(intent)
        
        # Technical domain detection (helps determine if general knowledge might be relevant)
        technical_domains = {
            'mobile': ['android', 'ios', 'mobile', 'app', 'kotlin', 'swift', 'react native'],
            'web': ['web', 'javascript', 'html', 'css', 'react', 'vue', 'angular', 'frontend', 'backend'],
            'api': ['api', 'rest', 'graphql', 'endpoint', 'request', 'response', 'http'],
            'database': ['database', 'sql', 'mongodb', 'postgres', 'mysql', 'query'],
            'payment': ['payment', 'billing', 'stripe', 'paypal', 'transaction', 'checkout'],
            'authentication': ['auth', 'login', 'oauth', 'jwt', 'token', 'security'],
            'integration': ['integration', 'webhook', 'sdk', 'plugin', 'connect']
        }
        
        detected_domains = []
        for domain, keywords in technical_domains.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_domains.append(domain)
        
        # Determine if this query likely needs implementation examples
        needs_examples = any(pattern in question_lower for pattern in [
            'how to', 'example', 'code', 'implement', 'integrate', 'tutorial', 'guide'
        ])
        
        return {
            'intents': detected_intents,
            'primary_intent': detected_intents[0] if detected_intents else 'general',
            'technical_domains': detected_domains,
            'needs_examples': needs_examples,
            'complexity': 'high' if len(detected_domains) > 1 or needs_examples else 'medium'
        }
    
    def enhance_context_with_analysis(self, question: str, context_docs: List) -> str:
        """
        Enhance the context provided to the LLM with query analysis and guidance.
        
        This method:
        1. Analyzes what the user is asking for
        2. Examines what information is available in documents
        3. Provides guidance to the LLM about when to use general knowledge
        
        Args:
            question (str): The user's question
            context_docs (List): Retrieved document chunks
            
        Returns:
            str: Enhanced context string with analysis and guidance
        """
        # Analyze the user's query
        query_analysis = self.analyze_query_intent(question)
        
        # Analyze document content for topic coverage
        doc_topics = set()
        doc_content_summary = []
        
        for doc in context_docs:
            content = doc.page_content.lower()
            
            # Extract topics mentioned in documents
            for domain, keywords in {
                'mobile': ['android', 'ios', 'mobile', 'app'],
                'web': ['web', 'javascript', 'html', 'css', 'react'],
                'api': ['api', 'rest', 'endpoint', 'request'],
                'database': ['database', 'sql', 'query'],
                'payment': ['payment', 'billing', 'transaction'],
                'authentication': ['auth', 'login', 'oauth', 'token'],
                'integration': ['integration', 'webhook', 'sdk']
            }.items():
                if any(keyword in content for keyword in keywords):
                    doc_topics.add(domain)
            
            # Create content summary
            doc_summary = {
                'source': doc.metadata.get('source', 'Unknown'),
                'has_code': 'code' in content or '```' in doc.page_content,
                'has_examples': 'example' in content or 'tutorial' in content,
                'length': len(doc.page_content),
                'key_topics': [topic for topic in doc_topics if any(
                    keyword in content for keyword in {
                        'mobile': ['android', 'ios'], 'web': ['javascript', 'react'],
                        'api': ['api', 'endpoint'], 'payment': ['payment', 'billing']
                    }.get(topic, [])
                )]
            }
            doc_content_summary.append(doc_summary)
        
        # Create enhanced context guidance
        context_guidance = f"""
QUERY ANALYSIS:
- Primary Intent: {query_analysis['primary_intent']}
- Technical Domains: {', '.join(query_analysis['technical_domains']) if query_analysis['technical_domains'] else 'General'}
- Needs Examples: {'Yes' if query_analysis['needs_examples'] else 'No'}
- Complexity: {query_analysis['complexity']}

DOCUMENT ANALYSIS:
- Topics Covered: {', '.join(doc_topics) if doc_topics else 'General content'}
- Total Chunks: {len(context_docs)}
- Has Code Examples: {any(doc['has_code'] for doc in doc_content_summary)}
- Has Tutorials: {any(doc['has_examples'] for doc in doc_content_summary)}

HYBRID RESPONSE GUIDANCE:
"""
        
        # Determine response strategy based on analysis
        if query_analysis['needs_examples'] and not any(doc['has_code'] for doc in doc_content_summary):
            if any(domain in doc_topics for domain in query_analysis['technical_domains']):
                context_guidance += """
✅ SUPPLEMENT WITH EXAMPLES: Documents mention the topic but lack implementation details.
   Provide relevant code examples and implementation guidance.
   Clearly mark what comes from documents vs. general knowledge.
"""
            else:
                context_guidance += """
❌ STAY DOCUMENT-FOCUSED: Topic not clearly covered in documents.
   Acknowledge limitation and suggest document-related questions.
"""
        elif len(context_docs) > 0:
            context_guidance += """
✅ DOCUMENT-BASED RESPONSE: Sufficient information available in documents.
   Focus on document content with minimal general knowledge supplementation.
"""
        else:
            context_guidance += """
❌ INSUFFICIENT CONTEXT: No relevant documents found.
   Politely decline and suggest questions about document content.
"""
        
        return context_guidance
    
    def get_enhanced_response(self, question: str) -> dict:
        """
        Get an enhanced response using hybrid approach with detailed analysis and token tracking.
        
        This method orchestrates the entire hybrid response process:
        1. Analyzes the query intent and complexity
        2. Retrieves relevant document context
        3. Enhances context with analysis and guidance
        4. Generates response with appropriate hybrid strategy
        5. Tracks token usage for cost monitoring and optimization
        
        Args:
            question (str): The user's question
            
        Returns:
            dict: Enhanced response with metadata, analysis, and token usage
        """
        print(f"🔍 Analyzing query: {question}")
        
        # Step 1: Analyze the query
        query_analysis = self.analyze_query_intent(question)
        print(f"📊 Query analysis: {query_analysis}")
        
        # Step 2: Get relevant context
        context_docs = self.get_relevant_context(question, k=6)
        print(f"📚 Retrieved {len(context_docs)} context documents")
        
        # Step 3: Enhance context with analysis
        context_guidance = self.enhance_context_with_analysis(question, context_docs)
        print(f"🎯 Generated context guidance for hybrid response")
        
        # Step 4: Calculate input token count for cost tracking
        # This helps users understand the cost and complexity of their queries
        input_text = question
        if context_docs:
            # Add context length to input calculation
            context_text = "\n".join([doc.page_content for doc in context_docs])
            input_text += f"\n{context_text}"
        
        # Rough token estimation (1 token ≈ 4 characters for most models)
        # This is an approximation since exact tokenization requires the model's tokenizer
        estimated_input_tokens = len(input_text) // 4
        
        print(f"📏 Estimated input tokens: {estimated_input_tokens}")
        
        # Step 5: Get response from QA chain and measure output
        response = self.qa_chain.invoke({"question": question})
        
        # Calculate output token count
        output_text = response['answer']
        estimated_output_tokens = len(output_text) // 4
        
        print(f"📏 Estimated output tokens: {estimated_output_tokens}")
        
        # Step 6: Calculate token usage statistics
        total_tokens = estimated_input_tokens + estimated_output_tokens
        
        # Estimate cost based on Google Gemini pricing (approximate)
        # Input: $0.00015 per 1K tokens, Output: $0.0006 per 1K tokens
        estimated_input_cost = (estimated_input_tokens / 1000) * 0.00015
        estimated_output_cost = (estimated_output_tokens / 1000) * 0.0006
        total_estimated_cost = estimated_input_cost + estimated_output_cost
        
        # Create detailed token usage information
        token_usage = {
            'input_tokens': estimated_input_tokens,
            'output_tokens': estimated_output_tokens,
            'total_tokens': total_tokens,
            'context_docs_count': len(context_docs),
            'context_length': sum(len(doc.page_content) for doc in context_docs),
            'question_length': len(question),
            'answer_length': len(output_text),
            'estimated_cost': {
                'input_cost_usd': round(estimated_input_cost, 6),
                'output_cost_usd': round(estimated_output_cost, 6),
                'total_cost_usd': round(total_estimated_cost, 6)
            },
            'efficiency_metrics': {
                'tokens_per_context_doc': round(estimated_input_tokens / max(len(context_docs), 1), 2),
                'output_input_ratio': round(estimated_output_tokens / max(estimated_input_tokens, 1), 2),
                'cost_per_response_cents': round(total_estimated_cost * 100, 4)
            }
        }
        
        print(f"💰 Token usage: {total_tokens} total ({estimated_input_tokens} in, {estimated_output_tokens} out)")
        print(f"💰 Estimated cost: ${total_estimated_cost:.6f} USD")
        
        # Step 7: Enhance response with analysis metadata and token tracking
        enhanced_response = {
            'answer': response['answer'],
            'source_documents': response.get('source_documents', []),
            'query_analysis': query_analysis,
            'context_guidance': context_guidance,
            'total_context_docs': len(context_docs),
            'token_usage': token_usage,  # Add comprehensive token tracking
            'response_metadata': {
                'has_code_examples': '```' in response['answer'],
                'response_length': len(response['answer']),
                'likely_hybrid': query_analysis['needs_examples'] and len(context_docs) > 0,
                'processing_timestamp': time.time()
            }
        }
        
        print(f"✅ Generated enhanced hybrid response with token tracking")
        return enhanced_response
    
    def enhance_query(self, question: str) -> str:
        """Enhance the query for better retrieval."""
        # Add context keywords based on common document patterns
        enhanced_question = question
        
        # Add relevant keywords for better matching
        if any(word in question.lower() for word in ['how', 'guide', 'tutorial', 'steps']):
            enhanced_question += " instructions process steps"
        
        if any(word in question.lower() for word in ['what', 'definition', 'meaning']):
            enhanced_question += " definition explanation overview"
            
        if any(word in question.lower() for word in ['integration', 'api', 'connect']):
            enhanced_question += " integration API technical implementation"
            
        if any(word in question.lower() for word in ['payment', 'billing', 'cost']):
            enhanced_question += " payment billing cost pricing"
            
        return enhanced_question
    
    def get_relevant_context(self, question: str, k: int = 6) -> List:
        """Get relevant context using multiple retrieval strategies."""
        if not self.vector_store:
            return []
        
        # Enhance the query
        enhanced_question = self.enhance_query(question)
        
        # Use multiple search strategies
        try:
            # 1. Similarity search
            similarity_docs = self.vector_store.similarity_search(enhanced_question, k=k//2)
            
            # 2. MMR search for diversity
            mmr_docs = self.vector_store.max_marginal_relevance_search(
                enhanced_question, 
                k=k//2, 
                fetch_k=k*2,
                lambda_mult=0.7
            )
            
            # Combine and deduplicate
            all_docs = similarity_docs + mmr_docs
            seen_content = set()
            unique_docs = []
            
            for doc in all_docs:
                content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            return unique_docs[:k]
            
        except Exception as e:
            print(f"Error in context retrieval: {e}")
            # Fallback to simple similarity search
            return self.vector_store.similarity_search(question, k=k)

def main():
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Try to load existing vector store
    if not rag.load_vector_store():
        # If no vector store exists, process documents
        documents = rag.load_documents()
        chunks = rag.process_documents(documents)
        rag.create_vector_store(chunks)
    
    # Set up the QA chain
    rag.setup_qa_chain()
    
    # Interactive query loop
    print("\nRAG System Ready! Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            break
        
        try:
            response = rag.get_enhanced_response(question)
            print(f"\nAnswer: {response['answer']}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 