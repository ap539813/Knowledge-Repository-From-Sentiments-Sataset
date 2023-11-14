# Import necessary modules for data visualization, web app building and data processing
import altair as alt
import streamlit as st

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

import requests

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



if not 'responses' in st.session_state:
    st.session_state['responses'] = {}


def main():

    for question_i, answer_i in st.session_state['responses'].items():
        col11, col12 = st.expander(question_i).columns([1, 1])
        col11.markdown('### Response from OpenAI API')
        col12.markdown('### Response from RAG')
        col11.markdown(answer_i[0])
        col12.markdown(answer_i[1])

    

    question_input = st.text_input("Ask your question: ")
    setting_cols = st.columns([1, 1, 1])
    tmperature = setting_cols[0].number_input("Choose temperature of the model: ")
    answer_type = setting_cols[1].selectbox('Choose Answer type: ', ["Long", "Short"])
    answer_button = st.button('Ask to VastuGPT')

    if answer_button:
        llm = st.session_state['GPT']
        template_for_baseline = """
            Answer question from Vastu Shashtra document:


            Question:
                {question}

            Instructions:
                1. Be very specific to the document I have in conversation history
                2. Do not refer to anything else external from the source
                3. Answer should be {answer_type} answer type

            Answer:
            """
        prompt_repo = PromptTemplate(
            template = template_for_baseline,
            input_variables=['question', 'answer_type'],
        )
        question = "What does Vastu say about kitchen placement direction answer in short"

        llm_chain = LLMChain(prompt=prompt_repo, llm=llm)
        answer = llm_chain.run(
            conversation_history = vastu_text,
            question = question_input,
            answer_type = answer_type
        )

        # answer_rag = query_rag("What does Vastu say about kitchen placement?", st.session_state['model'], st.session_state['vector_store'], st.session_state['document_chunks'])

        url = 'http://127.0.0.1:5000/'
        rag_endpoint = 'rag'
        score_endpoint = 'bart_score'
        data = {'question': question_input, 'answer_type': answer_type}

        answer_rag = requests.post(url + rag_endpoint, json=data).json()['answer_rag']

        template_for_rag_llm = """
            Answer question from Vastu Shashtra document:


            Question:
                {question}

            Instructions:
                1. Be very specific to the text I have in conversation history
                2. Do not refer to anything else external from the source
                3. Answer should be {answer_type} answer type

            Answer:
            """
        prompt_repo = PromptTemplate(
            template = template_for_rag_llm,
            input_variables=['question', 'answer_type']
        )

        llm_chain = LLMChain(prompt=prompt_repo, llm=llm)
        answer_rag_llm = llm_chain.run(
            conversation_history = answer_rag,
            question = question_input,
            answer_type = answer_type
        )

        st.session_state['responses'][question_input] = [answer, answer_rag_llm]
        col1, col2 = st.columns([1, 1])

        col1.markdown('### Response from OpenAI API')
        col2.markdown('### Response from RAG')

        col1.markdown(answer)
        col2.markdown(answer_rag_llm)

        data = {'answer_rag': answer_rag_llm, 'answer_openai': answer}

        
        bart_score = requests.post(url + score_endpoint, json=data).json()

        st.markdown('#### Comparison Score')
        st.write(bart_score)

    



# Entry point of the application
if __name__ == '__main__':
    main()