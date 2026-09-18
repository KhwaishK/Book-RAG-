import streamlit as st
from query import query_book

st.set_page_config(page_title="Book Q&A", page_icon="📖")

st.title("📖 Ask the Book")
st.caption("RAG-powered natural language query system with page citations")

book_id = st.sidebar.text_input("Book ID", value="my-book-v1")
top_k = st.sidebar.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)

question = st.text_input("Ask a question about the book:")

if st.button("Ask") and question.strip():
    with st.spinner("Searching the book and generating an answer..."):
        result = query_book(question, book_id, k=top_k)

    st.subheader("Answer")
    st.write(result["answer"])

    if result["pages_used"]:
        st.info(f"📌 Pages referenced: {', '.join(map(str, result['pages_used']))}")

    with st.expander("View retrieved source chunks"):
        for c in result["retrieved_chunks"]:
            st.markdown(f"**Page {c['page_number']}** (similarity: {c['similarity']:.3f})")
            st.write(c["content"])
            st.divider()
