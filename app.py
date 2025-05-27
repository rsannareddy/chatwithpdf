import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks.manager import get_openai_callback
import tempfile



# Set page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide"
)

def init_session_state():
    """Initialize session state variables"""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""

def process_pdf(pdf_file, openai_api_key):
    """Process PDF and create conversational chain"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load PDF
        with st.spinner("Loading PDF..."):
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()

        # Split text into chunks
        with st.spinner("Processing text..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)

        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(chunks, embeddings)

        # Create conversation chain
        with st.spinner("Setting up chat..."):
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                return_source_documents=True,
                verbose=True
            )

        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return conversation_chain, len(chunks)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, 0

def handle_user_input(user_question):
    """Handle user input and generate response"""
    if st.session_state.conversation is None:
        st.error("Please upload and process a PDF first!")
        return
    
    with st.spinner("Thinking..."):
        try:
            with get_openai_callback() as cb:
                response = st.session_state.conversation({
                    "question": user_question
                })
            
            # Store conversation history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": response["answer"],
                "source_documents": response.get("source_documents", []),
                "tokens_used": cb.total_tokens,
                "cost": cb.total_cost
            })
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

def display_chat_history():
    """Display chat history with styling"""
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        # User question
        with st.container():
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 10px 0;">
                <strong>🙋 You:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
        
        # Assistant answer
        with st.container():
            st.markdown(f"""
            <div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; margin: 10px 0;">
                <strong>🤖 Assistant:</strong> {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
        
        # Show sources and metadata in expander
        with st.expander(f"📖 Sources & Details (Chat {len(st.session_state.chat_history) - i})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Token Usage:**", chat.get('tokens_used', 'N/A'))
                st.write("**Cost:**", f"${chat.get('cost', 0):.4f}")
            
            with col2:
                st.write("**Sources:**", len(chat.get('source_documents', [])))
            
            # Show source documents
            if chat.get('source_documents'):
                st.write("**Relevant Document Sections:**")
                for j, doc in enumerate(chat['source_documents'][:2]):  # Show top 2 sources
                    st.write(f"**Source {j+1}:**")
                    st.write(f"Page: {doc.metadata.get('page', 'Unknown')}")
                    st.write(f"Content: {doc.page_content[:200]}...")
                    st.write("---")

def main():
    st.title("📚 PDF Chat Assistant")
    st.markdown("Upload a PDF and start chatting with your document!")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the service"
        )
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to continue.")
            st.info("You can get your API key from https://platform.openai.com/api-keys")
        
        st.divider()
        
        # PDF Upload
        st.header("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file and openai_api_key:
            if st.button("Process PDF", type="primary"):
                conversation_chain, num_chunks = process_pdf(uploaded_file, openai_api_key)
                
                if conversation_chain:
                    st.session_state.conversation = conversation_chain
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []  # Reset chat history
                    
                    st.success(f"✅ PDF processed successfully!")
                    st.info(f"Document split into {num_chunks} chunks")
        
        # Display current PDF info
        if st.session_state.pdf_processed:
            st.divider()
            st.header("Current Document")
            st.write(f"📄 **{st.session_state.pdf_name}**")
            st.write(f"💬 **Chats:** {len(st.session_state.chat_history)}")
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Main chat interface
    if st.session_state.pdf_processed and openai_api_key:
        st.header(f"💬 Chat with {st.session_state.pdf_name}")
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about your PDF:",
            placeholder="What is this document about?",
            key="user_input"
        )
        
        # Submit button or Enter key
        if st.button("Send", type="primary") or user_question:
            if user_question.strip():
                handle_user_input(user_question)
                st.rerun()
        
        # Sample questions
        st.markdown("**💡 Sample Questions:**")
        sample_questions = [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the conclusions or recommendations?",
            "Are there any specific dates or numbers mentioned?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{i}"):
                    handle_user_input(question)
                    st.rerun()
        
        st.divider()
        
        # Display chat history
        if st.session_state.chat_history:
            st.header("📝 Chat History")
            display_chat_history()
        else:
            st.info("👆 Ask a question to start chatting with your PDF!")
    
    elif not st.session_state.pdf_processed:
        # Welcome message
        st.markdown("""
        ## 🚀 How to get started:
        
        1. **Enter your OpenAI API Key** in the sidebar
        2. **Upload a PDF document** using the file uploader
        3. **Click 'Process PDF'** to analyze your document
        4. **Start chatting** with your document!
        
        ### ✨ Features:
        - **Smart chunking** for better context understanding
        - **Source citations** showing which parts of the PDF were used
        - **Conversation memory** to maintain context across questions
        - **Cost tracking** to monitor API usage
        - **Sample questions** to help you get started
        
        ### 🔒 Privacy:
        - Your PDF is processed locally and temporarily
        - Only text content is sent to OpenAI for processing
        - No documents are permanently stored
        """)

# Alternative version without Streamlit (Command Line Interface)
class PDFChatCLI:
    def __init__(self, pdf_path, openai_api_key):
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        self.conversation = None
        self.setup_conversation()
    
    def setup_conversation(self):
        """Set up the conversational chain"""
        try:
            # Load PDF
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Create conversation chain
            llm = ChatOpenAI(
                temperature=0,
                openai_api_key=self.openai_api_key
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            self.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=True
            )
            
            print(f"✅ PDF loaded and processed: {len(chunks)} chunks created")
            
        except Exception as e:
            print(f"❌ Error setting up conversation: {e}")
    
    def chat(self):
        """Start the chat loop"""
        if not self.conversation:
            print("❌ Conversation not set up properly.")
            return
        
        print("\n🤖 PDF Chat Assistant")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        while True:
            user_input = input("\n🙋 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("""
Available commands:
- quit/exit/q: Exit the chat
- help: Show this help message
- Just type your question to chat with the PDF
                """)
                continue
            elif not user_input:
                continue
            
            try:
                with get_openai_callback() as cb:
                    response = self.conversation({"question": user_input})
                
                print(f"\n🤖 Assistant: {response['answer']}")
                print(f"\n💰 Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
                
                # Show sources
                if response.get('source_documents'):
                    print(f"📖 Sources: {len(response['source_documents'])} document sections")
                
            except Exception as e:
                print(f"❌ Error: {e}")
