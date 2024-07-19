import streamlit as st
import importlib
import Rag_Utils
importlib.reload(Rag_Utils)

question = st.text_input("Enter your question")
st.write(question)
if question:
    query_engine = Rag_Utils.llama_Rag(question)
    answer = Rag_Utils.rag_query(question, query_engine)
    st.write_stream(answer.response_gen)