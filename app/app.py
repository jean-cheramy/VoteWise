import streamlit as st
from typing import List
from load_index import load_vector_store
from generator import generate_answer

st.title("VoteWise — Belgian Political Program Comparator")

vector_store = load_vector_store()

def ask_question(question: str) -> None:
    """
    Retrieve relevant documents from the vector store and generate a summarized answer.
    
    Args:
        question (str): The user's question.
    """
    with st.spinner("Retrieving..."):
        results = vector_store.similarity_search(question, k=5)
        contexts: List[str] = [r.page_content for r in results]
        answer: str = generate_answer(question, contexts)
    
    st.subheader("Answer")
    st.write(answer)
    
    st.subheader("Retrieved Contexts")
    for ctx in contexts:
        st.write("•", ctx)


question_input: str = st.text_input("Ask a question:", "Que proposent les partis wallons sur le climat ?")

if st.button("Ask"):
    ask_question(question_input)
