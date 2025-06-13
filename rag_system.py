import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import (
    TextLoader,
    # UnstructuredPDFLoader,
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

    def load_documents(self) -> List:
        """Load documents from the assets directory, dispatching by file extension."""
        loaders = {
            ".txt": (TextLoader, {"encoding": "utf-8"}),
            ".pdf": (PyPDFLoader, {"extract_images": False}),
            ".docx": (Docx2txtLoader, {}),
        }

        documents = []
        for root, _, files in os.walk(self.assets_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                entry = loaders.get(ext)
                if not entry:
                    continue  # skip unsupported types

                loader_cls, loader_kwargs = entry
                path = os.path.join(root, fn)
                try:
                    loader = loader_cls(path, **loader_kwargs)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading file '{path}': {e}")

        print(f"Loaded {len(documents)} documents")
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
        """Set up the question-answering chain with improved retrieval."""
        # Initialize the language model with a system prompt
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0.3,  # Lower temperature for more focused responses
            streaming=False,  # Disable streaming
            system_instruction="""You are a helpful AI assistant that provides accurate, well-formatted responses based on the provided context documents.

CRITICAL INSTRUCTIONS:
1. ONLY use information from the provided context documents
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Always cite which parts of the context you're using
4. Structure your response with clear markdown formatting
5. Be comprehensive but focused - don't add information not in the context

FORMATTING REQUIREMENTS:
1. Use clear headings (# ## ###) to organize information
2. Use bullet points (-) for lists and features
3. Use numbered lists (1. 2. 3.) for sequential steps
4. Use **bold** for important terms and concepts
5. Use `code formatting` for technical terms, file names, or commands
6. Add blank lines between sections for readability
7. Use > blockquotes for important notes or warnings

RESPONSE STRUCTURE:
- Start with a direct answer to the question
- Provide detailed explanation with proper formatting
- Include relevant examples from the context if available
- End with a summary if the topic is complex

Remember: Base your response ONLY on the provided context. If information is missing, acknowledge this limitation."""
        )
        
        # Set up memory for conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer" 
        )
        
        # Create a more sophisticated retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 6,  # Get more chunks initially
                "fetch_k": 12,  # Fetch more candidates for MMR
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        # Create the QA chain with custom prompt
        from langchain.prompts import PromptTemplate
        
        custom_prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            
Context information:
{context}

Previous conversation:
{chat_history}

Question: {question}

Instructions:
- Provide a comprehensive answer based ONLY on the context provided
- If the context doesn't contain sufficient information, clearly state this
- Use proper markdown formatting for better readability
- Structure your response logically with headings and bullet points
- Be specific and cite relevant parts of the context

Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=base_retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer",
            return_generated_question=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
    
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
            response = rag.qa_chain.invoke({"question": question})
            print(f"\nAnswer: {response['answer']}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 