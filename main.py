import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


from htmlTemplates import bot_template, user_template, css

import csv
import docx
import io

def get_pdf_text(pdf_files):
    
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_csv_text(csv_files):

    text = ""
    for csv_file in csv_files:
        decoded = csv_file.getvalue().decode('utf-8') 
        reader = csv.reader(io.StringIO(decoded)) 
        for row in reader:
            text += ' '.join(row) + '\n'
    return text

def get_word_text(word_files):

    text = ""
    for word_file in word_files:
        doc = docx.Document(word_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
    return text

def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    
    embeddings = OpenAIEmbeddings(openai_api_key='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')

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore

def get_conversation_chain(vector_store):

    llm = ChatOpenAI(openai_api_key='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')

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # DEFAULT_SYSTEM_PROMPT = """
    # You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    # """.strip()


    # def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    #     return f"""
    # [INST] <>
    # {system_prompt}
    # <>

    # {prompt} [/INST]
    # """.strip()

    # SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

    # template = generate_prompt(
    #     """
    # {context}

    # Question: {question}
    # """,
    #     system_prompt=SYSTEM_PROMPT,
    # )

    # prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory,
        # chain_type_kwargs={"prompt": prompt},
    )

    return conversation_chain

def handle_user_input(question):

    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def handle_file_input(file, file_type):
    if file_type == 'pdf':
        text = get_pdf_text([file])
    elif file_type == 'csv':
        text = get_csv_text([file])
    elif file_type == 'docx':
        text = get_word_text([file])
    else:
        text = ""

    return text

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Your own Files', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Chat with Your own Files :books:')
    question = st.text_input("Ask anything to your file: ")

    if question:
        handle_user_input(question)
    

    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        files = st.file_uploader("Choose your Files and Press OK", type=['pdf', 'csv', 'docx'], accept_multiple_files=True)

        if st.button("OK"):
            with st.spinner("Processing your Files..."):

                raw_text = ""
                for file in files:
                    file_type = file.name.split('.')[-1]
                    text = handle_file_input(file, file_type)
                    raw_text += text

                text_chunks = get_chunk_text(raw_text)
                
                vector_store = get_vector_store(text_chunks)
                st.write("DONE")

                st.session_state.conversation =  get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()