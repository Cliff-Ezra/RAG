# Import necessary modules
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import Replicate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load the .env file
load_dotenv()

# Access the environment variable
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
DATA_PATH = os.getenv('DATA_PATH')

result = {}

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
llama2_13b = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
llm = Replicate(
    model=llama2_13b,
    model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
)

# Set up ConversationBufferMemory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# Set up Streamlit interface
st.title('👨🏽‍⚖️ Law Guru')

# Display the number and names of the PDF files
pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
# Display the number of PDF files in a dropdown
with st.expander(f"Number of PDF files: {len(pdf_files)}"):
    st.write("PDF file names:")
    for file_name in pdf_files:
        st.write(file_name)


# Get user's question
question = st.text_input('Input your question here')

# Create a submit button
submit_button = st.button('Submit')

# Get and display model's answer when the submit button is pressed
if submit_button and question:
    # Load documents
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()

    # Create a text splitter to split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(docs)

    # create the vector db to store all the split chunks as embeddings
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
    )

    # Retrieve answers using the LLM and the vector db
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )

    # Save the previous question and answer to memory
    if 'last_question' in st.session_state and 'last_answer' in st.session_state:
        memory.save_context({"input": st.session_state.last_question}, {"output": st.session_state.last_answer})

    # Get the model's answer
    result = qa_chain({"query": question})

    # Check if result is a dictionary and contains 'result' key
    if isinstance(result, dict) and 'result' in result:
        # Display the answer
        st.write(result['result'])

        # Create a dropdown for the source text
        # with st.expander('Source Text'):
        #     sources = result.get('sources', 'No sources found')
        #     st.write(sources)

        # Save the current question and answer to the session state
        st.session_state.last_question = question
        st.session_state.last_answer = result['result']
    else:
        # If result is a string, display it directly
        st.write(result)
