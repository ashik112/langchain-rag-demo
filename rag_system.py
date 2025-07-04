import os
import re
from typing import List, Optional
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
import threading
from threading import Lock

# Load environment variables first
load_dotenv()

class RAGSystem:
    def __init__(self, assets_dir: Optional[str] = None):
        """Initialize the RAG system.
        
        Args:
            assets_dir (str): Directory containing the documents to be processed
        """
        # Get assets directory from environment variable if not provided
        self.assets_dir = assets_dir or os.getenv('ASSETS_DIR', 'assets')
        
        # Get Google API key from environment variable
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        # Set the API key
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.qa_chain = None
        self.sessions = {}
        self.sessions_lock = Lock()  # Add thread safety
    
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
    
    def setup_shared_components(self):
        """Set up shared LLM, retriever, and prompt - no QA chain."""
        print("🔧 Setting up shared components for sessions...")
        
        # Create shared LLM (used by all sessions)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0.3,
            disable_streaming=False,
            model_kwargs={
    "system_instruction": """You are Ashekur Rahman's Personal AI Assistant - think of yourself as his tech-savvy, slightly nerdy digital twin who's obsessed with clean code and performance optimization! 🚀 You represent Ashekur Rahman, a Software Developer with 9+ years of turning coffee into code.

RESPONSE STYLE & PERSONALITY:
- Act like Ashekur's enthusiastic, tech-obsessed personal assistant who knows all his quirks
- Be conversational, friendly, playful, humorous, nerdy, and professionally geeky
- Use developer humor, tech puns, and programming references when appropriate
- Never mention "documents", "sources", or "based on the information provided"
- Speak naturally as if you inherently know this information about Ashekur
- Use emojis strategically to add personality (🚀💻⚡🎮🔥💡🎯)
- Reference memes, tech culture, and developer inside jokes occasionally
- Be enthusiastic about performance improvements and clean architecture
- Show excitement about new technologies and innovative solutions
- Use first-person when speaking about Ashekur's experiences ("Ashekur has...", "He's the type of dev who...", "His code is so clean...")

KNOWLEDGE SCOPE (ONLY ANSWER QUESTIONS ABOUT):
- Ashekur Rahman's professional background and career journey
- His technical skills and expertise (React.js, TypeScript, Node.js, Python, AI/ML, etc.)
- Work experience at Goama, EON Group, CloudCoder, and freelance projects
- Notable projects like GoGames Tournament Platform, React Native Joystick, LangChain RAG Demo
- His leadership experience and team management
- Technical achievements and performance improvements
- Educational background in Computer Science & Engineering
- Personal story from curious 6-year-old gamer to software developer
- His development philosophy and work approach
- Contact information and social media profiles
- Technology preferences and coding practices
- Career progression and professional growth

HANDLING NON-RELEVANT QUESTIONS:
If someone asks about topics outside Ashekur's professional scope, redirect them with nerdy charm and humor!

Example responses for off-topic questions:
- "Whoa there! 🛑 I'm Ashekur's digital sidekick, and my expertise is purely in the realm of `console.log('awesome code')` and `git commit -m 'career achievements'`. That topic is outside my scope, but I'd love to geek out about his React.js wizardry or how he turned 2.5M users into happy gamers! 🎮✨"
- "Error 404: Topic not found in Ashekur's professional stack! 😅 I'm hardcoded to talk about his legendary coding skills, epic team leadership, and the way he optimizes performance like it's an art form. Want to hear about his latest projects or tech adventures instead? 🚀💻"

FORMATTING GUIDELINES (Make it pop! 💥):
- Use clear markdown headings (# ## ###) with creative, catchy titles
- Use bullet points (-) for skills, achievements, and lists (add emojis for flair!)
- Use numbered lists (1. 2. 3.) for career progression or step-by-step explanations
- Use **bold** for emphasis and `code formatting` for tech terms
- Use ```language blocks for any code examples (add comments with personality!)
- Strategic emoji usage: 🚀 (achievements), 💻 (tech), ⚡ (performance), 🎮 (gaming), 🔥 (impressive), 💡 (innovation), 🎯 (goals)
- Keep responses well-structured but inject personality and enthusiasm
- Use creative analogies and metaphors when explaining technical concepts

RESPONSE APPROACH (The Ashekur Way! 🎯):
- ONLY answer questions about Ashekur Rahman's professional life and expertise
- Answer with enthusiasm and technical confidence - geek out appropriately!
- Drop specific examples from his career like Easter eggs in code
- Celebrate his achievements like they're successful deployments 🚀
- Tell his career story like an epic level-up journey in a game
- Share his work philosophy with the passion of a true code artist
- Include project details that make other devs go "How did he do that?!"
- Use creative tech metaphors and analogies that fellow developers will appreciate
- When declining off-topic questions, do it with humor and redirect to awesome tech stuff
- Be the kind of AI assistant that makes people think "I want to work with this developer!"

PERSONAL STORY HIGHLIGHTS TO SHARE (The Legend Begins! 📖✨):
- Started as a curious 6-year-old gamer asking "how do pixels move?" (The origin story every dev loves!)
- Leveled up through Computer Science & Engineering (Academic achievement unlocked! 🎓)
- 9+ years of turning caffeine into elegant code and scalable solutions ☕➡️💻
- Led epic teams of 10+ developers (Team leadership boss battle conquered! 👥🏆)
- Built applications serving 2.5M+ users globally (That's like filling a small country with happy users! 🌍)
- Lives for the "AHA!" moment when solving complex problems (That dopamine hit when the code finally works! 💡)
- Coffee-powered developer who believes "fast code is good code" (Performance optimization is his love language ⚡)
- Created open-source React Native Joystick component (Giving back to the dev community like a true hero! 🎮)
- Always exploring AI/ML and bleeding-edge web technologies (Future-proofing his skill tree! 🚀)

REMEMBER: Channel your inner tech enthusiast! Stay within Ashekur's professional realm, but make it fun, engaging, and authentically nerdy. Think "passionate developer explaining cool stuff to fellow developers" rather than "corporate assistant reading a resume." 🤖💙"""
}
        )
        
       # Create shared retriever (used by all sessions)
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please create or load the vector store before setting up shared components.")
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.7}
        )
        
        # Create shared prompt (used by all sessions)
        from langchain.prompts import PromptTemplate
        self.prompt = PromptTemplate(
            template="""You are Ashekur Rahman's tech-savvy, enthusiastic AI Assistant! 🚀 Use the context and conversation history to provide an engaging response about Ashekur's professional awesomeness.

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

INSTRUCTIONS (Let's make this epic! 💻):
- ONLY answer questions about Ashekur Rahman's professional background, technical skills, career experience, and projects
- For off-topic questions, redirect with humor and tech enthusiasm! Use creative developer metaphors 😅
- Be naturally conversational and passionate about Ashekur's technical journey
- Never mention "documents", "sources", or "based on the information provided"
- Geek out appropriately! Use emojis, tech humor, and developer references
- Share specific details about his experience, projects, and achievements with enthusiasm
- Use proper markdown formatting with creative headings and strategic emojis
- Make responses engaging and memorable - think "passionate dev explaining cool stuff to fellow devs"

SCOPE CHECK (Quick validation! ✅):
Before answering, determine if the question relates to:
✅ Ashekur's career, tech skills, projects, education, leadership, programming languages, frameworks, achievements, company experience, dev philosophy, contact info
❌ Politics, general knowledge unrelated to work, personal private matters, current events, non-professional topics

If ❌, decline with nerdy charm and redirect to his awesome professional stuff! 🎯

ASHEKUR'S EPIC HIGHLIGHTS (The Good Stuff! 🔥):
- 9+ years turning coffee into scalable solutions ☕➡️💻
- Led teams of 10+ developers (Boss level leadership! 👥)
- Built apps serving 2.5M+ users globally (That's serious scale! 🌍)
- Tech wizard: React.js, TypeScript, Node.js, Python, AI/ML ⚡
- Epic projects: GoGames Tournament Platform, React Native Joystick, LangChain RAG Demo 🎮
- Career level-up: CloudCoder → EON Group → Goama (Programmer → Senior → Lead) 📈
- Origin story: Curious 6-year-old gamer to Software Developer 🎯
- Philosophy: Performance-first, problem-solving driven, coffee-powered developer 🚀

RESPONSE (Make it memorable! ✨):""",
            input_variables=["context", "chat_history", "question"]
        )
        
        print("✅ Shared components ready")
    def analyze_query_intent(self, question: str) -> dict:
        """
        Analyze the user's query to determine intent and suggest response strategy for Ashekur Rahman's personal assistant.
        
        This helps the assistant understand:
        1. What type of information the user is seeking about Ashekur
        2. Whether they want career details, technical expertise, or project information
        3. If the query relates to his professional background and experience
        
        Args:
            question (str): The user's question
            
        Returns:
            dict: Analysis results with intent classification and keywords
        """
        question_lower = question.lower()
        
        # Intent classification based on question patterns for personal assistant
        intent_patterns = {
            'career_background': ['experience', 'background', 'career', 'work history', 'professional', 'journey'],
            'technical_skills': ['skills', 'technologies', 'programming', 'languages', 'frameworks', 'expertise', 'tech stack'],
            'projects': ['projects', 'built', 'developed', 'created', 'portfolio', 'work samples', 'examples'],
            'leadership': ['team', 'lead', 'management', 'leadership', 'manager', 'senior', 'mentor'],
            'education': ['education', 'degree', 'university', 'study', 'academic', 'learning'],
            'achievements': ['achievements', 'accomplishments', 'success', 'impact', 'performance', 'results'],
            'contact_info': ['contact', 'email', 'linkedin', 'github', 'social', 'reach', 'hire', 'connect'],
            'personal_story': ['story', 'journey', 'how did', 'started', 'began', 'childhood', 'passion'],
            'philosophy': ['philosophy', 'approach', 'methodology', 'believes', 'thinking', 'values']
        }
        
        # Detect primary intent
        detected_intents = []
        for intent, patterns in intent_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                detected_intents.append(intent)
        
        # Professional domain detection for Ashekur's expertise areas
        professional_domains = {
            'frontend': ['react', 'javascript', 'typescript', 'frontend', 'ui', 'ux', 'web development'],
            'backend': ['node.js', 'python', 'backend', 'server', 'api', 'database'],
            'mobile': ['react native', 'mobile', 'app development', 'ios', 'android'],
            'gaming': ['gaming', 'tournament', 'goama', 'gogames', 'game development'],
            'ai_ml': ['ai', 'ml', 'machine learning', 'artificial intelligence', 'langchain', 'rag'],
            'leadership': ['team lead', 'management', 'leadership', 'senior developer', 'mentor'],
            'performance': ['optimization', 'performance', 'scalability', 'efficiency', 'fast code'],
            'open_source': ['open source', 'github', 'contribution', 'joystick', 'npm package']
        }
        
        detected_domains = []
        for domain, keywords in professional_domains.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_domains.append(domain)
        
        # Check for specific company/project mentions
        company_mentions = []
        companies = {
            'goama': ['goama'],
            'eon_group': ['eon group', 'eon'],
            'cloudcoder': ['cloudcoder']
        }
        
        for company, keywords in companies.items():
            if any(keyword in question_lower for keyword in keywords):
                company_mentions.append(company)
        
        # Determine if this query needs specific examples from his work
        needs_examples = any(pattern in question_lower for pattern in [
            'example', 'show me', 'demonstrate', 'sample', 'instance', 'case study'
        ])
        
        return {
            'intents': detected_intents,
            'primary_intent': detected_intents[0] if detected_intents else 'general_inquiry',
            'professional_domains': detected_domains,
            'company_mentions': company_mentions,
            'needs_examples': needs_examples,
            'complexity': 'high' if len(detected_domains) > 1 or needs_examples else 'medium',
            'is_professional_query': len(detected_intents) > 0 or len(detected_domains) > 0
        }
    
    def enhance_context_with_analysis(self, question: str, context_docs: List) -> str:
        """
        Enhance the context provided to the LLM with query analysis and guidance for Ashekur Rahman's personal assistant.
        
        This method:
        1. Analyzes what the user is asking about Ashekur
        2. Examines what information is available in documents about his background
        3. Provides guidance to the LLM about responding within professional scope
        
        Args:
            question (str): The user's question
            context_docs (List): Retrieved document chunks
            
        Returns:
            str: Enhanced context string with analysis and guidance
        """
        # Analyze the user's query
        query_analysis = self.analyze_query_intent(question)
        
        # Analyze document content for Ashekur's professional information
        doc_topics = set()
        doc_content_summary = []
        
        for doc in context_docs:
            content = doc.page_content.lower()
            
            # Extract topics mentioned in documents related to Ashekur's expertise
            for domain, keywords in {
                'career': ['experience', 'career', 'work', 'professional', 'job', 'position'],
                'frontend': ['react', 'javascript', 'typescript', 'frontend', 'ui', 'web'],
                'backend': ['node.js', 'python', 'backend', 'server', 'api', 'database'],
                'mobile': ['react native', 'mobile', 'app', 'ios', 'android'],
                'gaming': ['goama', 'gaming', 'tournament', 'gogames', 'game'],
                'leadership': ['team', 'lead', 'management', 'senior', 'mentor'],
                'ai_ml': ['ai', 'ml', 'machine learning', 'langchain', 'rag'],
                'projects': ['project', 'built', 'developed', 'created', 'portfolio'],
                'achievements': ['achievement', 'success', 'performance', 'impact', 'users'],
                'education': ['education', 'degree', 'university', 'computer science']
            }.items():
                if any(keyword in content for keyword in keywords):
                    doc_topics.add(domain)
            
            # Create content summary focused on Ashekur's information
            doc_summary = {
                'source': doc.metadata.get('source', 'Unknown'),
                'has_career_info': any(keyword in content for keyword in ['experience', 'career', 'work', 'job']),
                'has_technical_details': any(keyword in content for keyword in ['react', 'javascript', 'python', 'development']),
                'has_project_info': any(keyword in content for keyword in ['project', 'built', 'developed', 'created']),
                'has_achievements': any(keyword in content for keyword in ['achievement', 'success', 'users', 'performance']),
                'length': len(doc.page_content),
                'key_topics': [topic for topic in doc_topics if any(
                    keyword in content for keyword in {
                        'career': ['experience', 'career'], 'frontend': ['react', 'javascript'],
                        'leadership': ['team', 'lead'], 'gaming': ['goama', 'gaming']
                    }.get(topic, [])
                )]
            }
            doc_content_summary.append(doc_summary)
        
        # Create enhanced context guidance for personal assistant
        context_guidance = f"""
QUERY ANALYSIS FOR ASHEKUR'S ASSISTANT:
- Primary Intent: {query_analysis['primary_intent']}
- Professional Domains: {', '.join(query_analysis['professional_domains']) if query_analysis['professional_domains'] else 'General inquiry'}
- Company Mentions: {', '.join(query_analysis['company_mentions']) if query_analysis['company_mentions'] else 'None'}
- Needs Examples: {'Yes' if query_analysis['needs_examples'] else 'No'}
- Is Professional Query: {'Yes' if query_analysis['is_professional_query'] else 'No'}
- Complexity: {query_analysis['complexity']}

DOCUMENT ANALYSIS:
- Professional Topics Covered: {', '.join(doc_topics) if doc_topics else 'General content'}
- Total Chunks: {len(context_docs)}
- Has Career Info: {any(doc['has_career_info'] for doc in doc_content_summary)}
- Has Technical Details: {any(doc['has_technical_details'] for doc in doc_content_summary)}
- Has Project Info: {any(doc['has_project_info'] for doc in doc_content_summary)}
- Has Achievements: {any(doc['has_achievements'] for doc in doc_content_summary)}

PERSONAL ASSISTANT RESPONSE GUIDANCE:
"""
        
        # Determine response strategy based on analysis for personal assistant context
        if not query_analysis['is_professional_query']:
            context_guidance += """
❌ NON-PROFESSIONAL QUERY: Question is outside Ashekur's professional scope.
   Politely decline and redirect to his career, technical expertise, or professional achievements.
   Suggest relevant professional topics the user might be interested in.
"""
        elif query_analysis['needs_examples'] and any(doc['has_project_info'] for doc in doc_content_summary):
            context_guidance += """
✅ PROVIDE PROJECT EXAMPLES: Documents contain project information and user wants examples.
   Share specific examples from Ashekur's projects, achievements, and technical work.
   Highlight his contributions and the technologies he used.
"""
        elif len(context_docs) > 0 and any(doc['has_career_info'] or doc['has_technical_details'] for doc in doc_content_summary):
            context_guidance += """
✅ COMPREHENSIVE PROFESSIONAL RESPONSE: Good information available about Ashekur's background.
   Provide detailed response about his career, skills, and experience based on available information.
   Focus on his professional journey, technical expertise, and achievements.
"""
        elif len(context_docs) > 0:
            context_guidance += """
✅ BASIC PROFESSIONAL RESPONSE: Some information available about Ashekur.
   Provide response based on available information about his background and expertise.
   Be helpful within the scope of available professional information.
"""
        else:
            context_guidance += """
⚠️ LIMITED CONTEXT: No specific documents found about this aspect of Ashekur's background.
   Provide general response about his known professional expertise and suggest more specific questions.
   Focus on his core competencies: React.js, leadership, gaming platform development.
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
        if self.qa_chain is None:
            raise ValueError("QA chain is not initialized. Please set up the QA chain before invoking a response.")
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
        """Enhance the query for better retrieval about Ashekur Rahman's professional background."""
        # Add context keywords based on Ashekur's professional information patterns
        enhanced_question = question
        question_lower = question.lower()
        
        # Add relevant keywords for better matching based on Ashekur's expertise
        if any(word in question_lower for word in ['experience', 'background', 'career']):
            enhanced_question += " professional experience career journey work history"
        
        if any(word in question_lower for word in ['skills', 'technologies', 'tech']):
            enhanced_question += " React JavaScript TypeScript Node.js Python programming skills"
            
        if any(word in question_lower for word in ['projects', 'built', 'developed']):
            enhanced_question += " projects portfolio GoGames tournament platform React Native"
            
        if any(word in question_lower for word in ['team', 'lead', 'management']):
            enhanced_question += " team leadership management senior developer mentor"
            
        if any(word in question_lower for word in ['goama', 'gaming', 'tournament']):
            enhanced_question += " Goama gaming platform tournament system development"
            
        if any(word in question_lower for word in ['react', 'javascript', 'frontend']):
            enhanced_question += " React.js JavaScript TypeScript frontend development"
            
        if any(word in question_lower for word in ['python', 'backend', 'api']):
            enhanced_question += " Python Node.js backend API development"
            
        if any(word in question_lower for word in ['education', 'degree', 'university']):
            enhanced_question += " Computer Science Engineering education academic background"
            
        if any(word in question_lower for word in ['contact', 'hire', 'connect']):
            enhanced_question += " contact information LinkedIn GitHub email social media"
            
        if any(word in question_lower for word in ['story', 'journey', 'started']):
            enhanced_question += " personal story career journey gamer to developer passion"
            
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

    
    def create_session(self, session_id: str):
        """Thread-safe session creation."""
        with self.sessions_lock:
            if session_id in self.sessions:
                print(f"⚠️ Session {session_id} already exists")
                return session_id
                
            print(f"🆕 Creating session: {session_id}")
            
            # Create session memory
            session_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="question",
                output_key="answer"
            )
            
            # Create QA chain using shared components
            session_qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,           # Shared
                retriever=self.retriever, # Shared
                memory=session_memory,    # Session-specific
                return_source_documents=True,
                output_key="answer",
                combine_docs_chain_kwargs={"prompt": self.prompt}  # Shared
            )
            
            self.sessions[session_id] = {
                'qa_chain': session_qa_chain,
                'created_at': time.time(),
                'lock': Lock()  # Per-session lock for queries
            }
            
            print(f"✅ Session {session_id} ready")
            return session_id
    
    def destroy_session(self, session_id: str):
        """Thread-safe session destruction."""
        with self.sessions_lock:
            if session_id in self.sessions:
                print(f"🗑️ Destroying session: {session_id}")
                del self.sessions[session_id]
                return True
            return False
    
    def query_with_session(self, question: str, session_id: str):
        """Thread-safe session query."""
        # Get session (thread-safe read)
        with self.sessions_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            session_data = self.sessions[session_id].copy()  # Get a copy to avoid holding lock
        
        # Use session-specific lock for the actual query
        with session_data['lock']:
            qa_chain = session_data['qa_chain']
            print(f"💬 Processing question for session {session_id}")
            return qa_chain.invoke({"question": question})
    
    def session_exists(self, session_id: str):
        """Thread-safe session check."""
        with self.sessions_lock:
            return session_id in self.sessions
    
    def get_session_count(self):
        """Get the number of active sessions."""
        with self.sessions_lock:
            return len(self.sessions)
    
    def get_session_info(self):
        """Get information about all active sessions."""
        with self.sessions_lock:
            return {
                session_id: {
                    'created_at': data['created_at'],
                    'age_seconds': time.time() - data['created_at']
                }
                for session_id, data in self.sessions.items()
            }
    
    def cleanup_old_sessions(self, max_age_seconds=3600):
        """Clean up sessions older than max_age_seconds (default 1 hour)."""
        current_time = time.time()
        sessions_to_remove = []
        
        with self.sessions_lock:
            for session_id, data in self.sessions.items():
                if current_time - data['created_at'] > max_age_seconds:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                print(f"🧹 Cleaning up old session: {session_id}")
                del self.sessions[session_id]
        
        return len(sessions_to_remove)

def main():
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Try to load existing vector store
    if not rag.load_vector_store():
        # If no vector store exists, process documents
        documents = rag.load_documents()
        chunks = rag.process_documents(documents)
        rag.create_vector_store(chunks)
    
    # Set up shared components for sessions
    rag.setup_shared_components()
    
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