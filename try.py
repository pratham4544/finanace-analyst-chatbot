import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate, LLMChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain



# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google API key
GoogleGenerativeAIEmbeddings.api_key = api_key

# Function to load and split the text from a PDF file
def get_file_text():
    text = ""
    pdf_reader = PdfReader('data/annual-report-2022-2023.pdf')
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to load or create a vector store from the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are finacical analyst having 10+ years in finace domain and you have provided the 
    annual report of TCS company now you have to understand the question and based on the
    context you have to provide the answer.
    Read the question {question} and find the correct verse from the context.
  
    Context:\n{context}\n
    Answer:
    """

    # Define the language model
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)

    # Create a prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Create the conversational chain
    return LLMChain(llm=model, prompt=prompt)

# Function to predict the next question
def predict_next_question(user_question):
    # Create a prompt template for predicting the next question
    prompt_template = """
    Based on the user's question: {user_question}, predict the next question the user might ask.
    """

    # Define the language model
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    
    # Create the prompt and LLM chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_question"])
    chain = LLMChain(llm=model, prompt=prompt)
    
    # Use the chain to predict the next question
    next_question = chain.run(user_question)
    
    return next_question

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # i comment this out because i don't want to use much resources 

    # text = get_file_text()


    # chunks = get_text_chunks(text=text)

    # db = get_vector_store(text_chunks=chunks)

    # Load the vector store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform a similarity search and retrieve relevant documents as context
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Run the chain with the retrieved context and the user's question
    response = chain.run({"context": docs, "question": user_question})

    # Create a container to display the response
    with st.container():
        st.write("Reply: ", response)

    # Predict the next question
    next_question = predict_next_question(user_question)
    
    # Display the predicted next question in the sidebar
    st.sidebar.header("Smart Suggestions")
    st.sidebar.write(next_question)

# Main function
def main():
    st.set_page_config("ReportBot💼")
    st.header("Find Insights in Annual Reports with Ease🏛️")
    st.subheader('How can I help you with the report?🧐')

    # User input
    user_question = st.text_input("Ask a Question ")

    if st.button("Submit"):
        if user_question:
            user_input(user_question)

if __name__ == "__main__":
    main()
