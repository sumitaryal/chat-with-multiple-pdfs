# Chat with Multiple PDFs using RAG

## Introduction 
The project showcases a system where users can upload their pdf and can ask questions related to the pdf which will be answered by the chatbot. 
The PDF is first read and then processed. It is first chunked down into multiple chunk of texts, which are then converted to vector embeddings and stored in a vector database. The user then asks a question based on the pdf contents. The question is converted into a question embedding and then semantic search is used to generate the ranked results from the vector store, which is then passed to the LLM to generate the answer of the question asked by the user based on the PDFs.

## Getting Started
1. Clone this repository:
    ```
    git clone https://github.com/sumitaryal/chat-with-multiple-pdfs.git
    ```
2. Create a virtual environment:
    ```
    python -m venv /path/to/new/virtual/environment
    ```
3. Install all the required packages:
    ```
    pip install -r requirements.txt
    ```
4. Setup the environment variables in the **.env** file as HUGGINGFACEHUB_API_TOKEN
   The token should be write-based token.
5. Run the project:
    ```
    python app.py
    ```

Here is the snippet of the UI 

![UI](/assets/UI.png)