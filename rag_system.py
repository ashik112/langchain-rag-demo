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
            ".docx": (Docx2txtLoader, {"extract_images": False}),
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
        """Process and chunk the documents."""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
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
        """Set up the question-answering chain."""
        # Initialize the language model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0.7,
            streaming=True  # Enable streaming
        )
        
        # Set up memory for conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",     # matches the chain's input
            output_key="answer" 
        )
        
        # Create the QA chain with streaming
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            output_key="answer",
            return_generated_question=True
        )
    
    def stream_query(self, question: str):
        """Stream a query response from the RAG system."""
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        
        print(f"Starting stream query for: {question}")
        
        # Get the sources first
        docs = self.vector_store.similarity_search(question, k=3)
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        print(f"Found sources: {sources}")
        
        try:
            # Get the full response first
            response = self.qa_chain.invoke({"question": question})
            answer = response["answer"]
            
            # Stream the response word by word
            words = answer.split()
            for i, word in enumerate(words):
                # Add a space after each word except the last one
                yield word + (" " if i < len(words) - 1 else "")
                # Small delay to simulate streaming
                time.sleep(0.05)
            
        except Exception as e:
            print(f"Error in stream_query: {str(e)}")
            raise
        
        # Yield the sources at the end
        sources_text = f"\n\nSources: {', '.join(set(sources))}"
        print(f"Yielding sources: {sources_text}")
        yield sources_text

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
            for token in rag.stream_query(question):
                print(token, end='', flush=True)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 