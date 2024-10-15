## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title('Celebrity Search Results')
input_text=st.text_input("Search the topic u want")

# Prompt Templates

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)


# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
occupation_memory = ConversationBufferMemory(input_key='person', memory_key='occupation_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')
wife_memory = ConversationBufferMemory(input_key='person', memory_key='wife_history')

## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

# Prompt Templates

first1_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="Tell me occupation of celebrity {person}"
)
chain1=LLMChain(
    llm=llm,prompt=first1_input_prompt,verbose=True,output_key='occupation',memory=occupation_memory)


second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)
# Prompt Templates

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)

fourth_input_prompt=PromptTemplate(
    input_variables=['person'],
    template=" His wife Name is{person}"
)
chain4=LLMChain(llm=llm,prompt=fourth_input_prompt,verbose=True,output_key='wife',memory=wife_memory) 

parent_chain=SequentialChain(
    chains=[chain,chain1,chain2,chain3,chain4],input_variables=['name'],output_variables=['person','occupation','dob','description','wife'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Occupation'): 
        st.info(occupation_memory.buffer)

    with st.expander('Wife Name'): 
        st.info(wife_memory.buffer)
      
    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)