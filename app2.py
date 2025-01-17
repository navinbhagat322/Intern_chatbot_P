#Detecting The History Too

import streamlit as st 
from langchain_ollama import ChatOllama #it will ensure that langchain framework can access our llama model
from langchain_core.output_parsers import StrOutputParser #Our output as parsed as string data

from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

#Typs of Prompts 
#1) system prompts 2) Human Prompt 3) Ai Prompt 
from langchain_core.prompts import ChatPromptTemplate #combing all of these prompt in a single chat template 


st.title("🤖 Chatbot For Conference Call Transcript")
st.write("This Chatbot is made using Ollama and Langchain")


#Now let's Build the model 
   #giving connection to laama  model using chatollama 
model = ChatOllama(model='llama3.2:3b',base_url='http://localhost:11434/')

#creating system messege template
system_message = SystemMessagePromptTemplate.from_template('yoy are a helful Ai Assitant.')
#it is not necessary to give system message but we have to give it so that LLm can know what's his role is






if "chat_history" not in st.session_state: #for each session , streamlit is going to maintain a session state. 
    st.session_state["chat_history"]=[]  #for that session state i am going to add my chat history. so that my LLM can get context from my previous question.


with st.form('llm-form'):
    text=st.text_area('Enter your text')
    submit = st.form_submit_button("Submit")

#method to generate the response
def generate_response(chat_history): #chat history have SystemMessagePromptTemplate,HumanMessegePromptTemplate, AIMessegePromptTemplate these messeges together
    chat_template = ChatPromptTemplate.from_messages(chat_history)
    chain = chat_template|model|StrOutputParser() #creating LLM chain 
#chat_template - langchian runable sequence - the task(chat prompt template )
# output from chatprompt template goes to the model and the output is coming to strOutputparser then output is coming like string value output 
#now we have to get the response 

    response = chain.invoke({}) #we have to create a empty dictonary becasue in starting our chat is empty
    return response  

#store user messege in user key and ai messegae in ai key
def get_history():
    chat_history =[system_message]
    for chat in st.session_state['chat_history']:
        prompt = HumanMessagePromptTemplate.from_template(chat['user'])
        chat_history.append(prompt)
        ai_message =AIMessagePromptTemplate.from_template(chat['assistant'])
        chat_history.append(ai_message)

    return chat_history


if submit and text:
    with st.spinner('Response is Cooking....'):
        #creating human messege template
        prompt = HumanMessagePromptTemplate.from_template(text)     

        chat_history = get_history() #getting chathistory from get history
        chat_history.append(prompt)
        response = generate_response(chat_history)
        st.session_state['chat_history'].append({'user': text, 'assistant':response}) #in the chat history we have list and inside the list we have item in the form of dictionary 
        #i am doing this because whenever i am doing new conversation so our llm should know the previous context or history
        #hence, basically we are appending our all set of conversation(human question and ai response ) 

        # st.write('response:',response)
        # st.write(st.session_state['chat_history'])

st.write('## Chat History')
for chat in reversed(st.session_state['chat_history']):
    st.write(f'**🧑‍🦰User**: {chat["user"]}')
    st.write(f'**🤖Assistant**: {chat["assistant"]}')
    st.write('---')