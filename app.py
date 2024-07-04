import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# app config
st.set_page_config(page_title="Chat Bot", page_icon="🤖")
st.title("ChatBot")


def get_Response(user_query, chat_history):
    """
    Return a response based on the given user query and chat history.
    """

    template = """
    You are a super helpful python programmer. Your mission is to code review other programmers code :

    Chat history: {chat_history}

    User question: {user_question}
    
    In bullet points, correct what's wrong in the code, or approve the code.
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = llm = Ollama(
        model="stablelm-zephyr",
        #model="codellama",
        base_url="http://localhost:11434",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am super helpful programmer assistant. How can I help you today ?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_Response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))



