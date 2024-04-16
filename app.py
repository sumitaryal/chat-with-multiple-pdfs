import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint


def get_text_from_pdf(pdf_docs):
    """
    Function to extract text from PDF files.
    """

    text = ""

    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(raw_text):
    """
    Function to split raw text into smaller chunks.
    """

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(raw_text)

    return chunks


def get_vector_store(text_chunks):
    """
    Function to create a vector store from text chunks.
    """

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vector_store


def get_conversation_chain(vector_store):
    """
    Function to create a conversation chain.
    """

    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.1", temperature=0.5)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain


def handle_user_input(user_question):
    """
    Function to handle user input and provide responses.
    """

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(response["chat_history"]):
        if i%2 == 0:
            st.chat_message("user").markdown(message.content)
        else:
            st.chat_message("assistant").markdown(message.content)


def main():
    """
    Main function to run the Streamlit application.
    """

    load_dotenv()

    st.set_page_config(page_title="Multi PDF Chatbot", page_icon="🤖")

    # Initialize session state variables if not already present
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Streamlit UI
    st.header("Multi PDF Chatbot 🤖")
    
    prompt = st.chat_input("Ask a question based on the PDF content:")
    if prompt:
        handle_user_input(prompt)


    # Sidebar section for PDF processing
    with st.sidebar:

        st.subheader("PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF files here and click on 'Process'", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):

            with st.spinner("Processing PDF files..."):

                # Get the text from the PDF files
                raw_text = get_text_from_pdf(pdf_docs)
                
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vector_store = get_vector_store(text_chunks)

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vector_store)    
    


if __name__ == "__main__":
    main()