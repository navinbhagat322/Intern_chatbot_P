import streamlit as st
from langchain_ollama import ChatOllama


st.title("🤖 Chatbot For Conference Call Transcript")
st.write("This Chatbot is made using Ollama and Langchain")

with st.form('llm-form'):
    text=st.text_area('Enter your text')
    submit = st.form_submit_button("Submit")

    #i am gonna write here method to generate Response

def generate_response(input_text):
    model = ChatOllama( model='llama3.2:3b')
    response = model.invoke(input_text)
    return response.content

if 'chat_history' not in st.session_state:
    st.session_state['chat_history']=[] #making a list for appending chat history


if submit and text:
    with st.spinner('Cooking Response...'):
        response =generate_response(text)
        st.session_state['chat_history'].append({'user': text,'ollama':response}) # we are appending our text data and user response
        st.write(response)

st.write('## Chat History')
for chat in reversed(st.session_state['chat_history']):
    st.write(f'**🧑‍🦰User**: {chat["user"]}')
    st.write(f'**🤖Assistant**: {chat["ollama"]}')
    st.write('.....')
