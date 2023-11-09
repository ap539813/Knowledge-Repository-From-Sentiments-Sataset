# Import necessary modules for data visualization, web app building and data processing
import altair as alt
import streamlit as st

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

# from sentence_transformers import SentenceTransformer

# from sklearn.neighbors import NearestNeighbors


from docx import Document

import json
import os


# Import the chat model
from langchain.chat_models import ChatOpenAI

# Set the page layout of the Streamlit app to be wide
st.set_page_config(layout="wide")

# Set the Altair visualization render options
alt.renderers.set_embed_options(scaleFactor=2)

with open('api_keys.json') as api_file:
    api_dict = json.load(api_file)

os.environ['OPENAI_API_KEY'] = api_dict['API OPENAI']

template_for_long_answer = """
Answer question from Vastu Shashtra document, The asnwer should be from the provided document only


Question:
    {question}

Instructions:
    1. Be very specific to the document I have in conversation history
    2. Do not refer to anything else external from the source

Answer:
"""

# Set the GPT model in the session state
st.session_state['GPT'] = OpenAI() #ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-16k")

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

vastu_text = extract_text_from_docx('vastu-shastra-processed.docx')
# We need to clean the document for un-necessary symbols so that the api calls can be optimised
def clean_text(text):
    cleaned_text = ''
    for char in text:
        if char.isalnum() or char.isspace() or char == '.':
            cleaned_text += char
    return cleaned_text

vastu_text = clean_text(vastu_text)

# def chunk_by_sentence(text, max_chunk_length=200):
#     paragraphs = text.split(".")  # Assuming sentences are separated by full stop.
#     chunks = []

#     for paragraph in paragraphs:
#         if len(paragraph) <= max_chunk_length:
#             chunks.append(paragraph)
#         else:
#             # For long sentences, further split them into smaller chunks
#             words = paragraph.split()
#             for i in range(0, len(words), max_chunk_length):
#                 chunk = " ".join(words[i:i+max_chunk_length])
#                 chunks.append(chunk)
                
#     return chunks



# def query_rag(question, model, vector_store, document_chunks):
#     question_embedding = model.encode([question])
#     _, indices = vector_store.kneighbors(question_embedding)
#     return document_chunks[indices[0][0]]




if not 'responses' in st.session_state:
    st.session_state['responses'] = {}
    # st.session_state['document_chunks'] = chunk_by_sentence(vastu_text)

    # st.session_state['model'] = SentenceTransformer('all-MiniLM-L6-v2')
    # st.session_state['embeddings'] = st.session_state['model'].encode(st.session_state['document_chunks'])

    # st.session_state['vector_store'] = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    # st.session_state['vector_store'].fit(st.session_state['embeddings'])

def main():
    for question_i, answer_i in st.session_state['responses'].items():
        st.expander(question_i).markdown(answer_i)

    question_input = st.text_input("Ask your question: ")
    answer_button = st.button('Ask to VastuGPT')

    if answer_button:
        llm = st.session_state['GPT']
        template_for_code = """
            Answer question from Vastu Shashtra document:


            Question:
                {question}

            Instructions:
                1. Be very specific to the document I have in conversation history
                2. Do not refer to anything else external from the source

            Answer:
            """
        prompt_repo = PromptTemplate(
            template = template_for_code,
            input_variables=['question']
        )
        question = "What does Vastu say about kitchen placement direction answer in short"

        llm_chain = LLMChain(prompt=prompt_repo, llm=llm)
        answer = llm_chain.run(
            conversation_history = vastu_text,
            question = question_input,
        )

        # answer_rag = query_rag("What does Vastu say about kitchen placement?", st.session_state['model'], st.session_state['vector_store'], st.session_state['document_chunks'])

        st.session_state['responses'][question_input] = answer
        # col1, col2 = st.columns([1, 1])

        # col1.markdown('### Response from OpenAI API')
        # col2.markdown('### Response from RAG')

        st.markdown(answer)
        # col2.markdown(answer_rag)

    



# Entry point of the application
if __name__ == '__main__':
    main()