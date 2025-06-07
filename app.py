import streamlit as st
from RagQA import RagQA
from summarizer import Summarizer


# Configuration
st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("📚 AI Document Assistant")

# ---------------- Session defaults ----------------------
if "success"       not in st.session_state: st.session_state.success       = False
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "model"         not in st.session_state: st.session_state.model         = None
if "rag"           not in st.session_state: st.session_state.rag           = None
if "doc_manager"   not in st.session_state: st.session_state.doc_manager   = None
if "summarizer"    not in st.session_state: st.session_state.summarizer    = None

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model = st.selectbox("Choose model", ["llama3.1:8b"])
    if st.session_state.model != model or st.session_state.rag is None:
        st.session_state.model       = model
        st.session_state.rag         = RagQA(model_name=model)
        st.session_state.doc_manager = st.session_state.rag.document_manager
        st.session_state.summarizer  = Summarizer(model_name=model)

    rag          = st.session_state.rag
    doc_manager  = st.session_state.doc_manager
    summarizer   = st.session_state.summarizer

    st.markdown("---")
    menu = st.radio(
        "🧭 Navigation",
        ["🏠 Home", "📥 Upload Documents", "💬 Chat", "📝 Summarize"]
    )

# HOME
if menu == "🏠 Home":
    st.markdown("## 👋 Welcome!")
    st.markdown("""
    This is an interactive application powered by RAG (Retrieval-Augmented Generation).

    - 📄 Upload documents
    - 💬 Ask questions based on content
    - ✍️ Generate summaries

    Choose an option from the menu on the left.
    """)

# UPLOAD DOCUMENTS
elif menu == "📥 Upload Documents":
    st.markdown("## 📥 Upload text or PDF files")
    uploaded_files = st.file_uploader("Drag and drop files:", accept_multiple_files=True, type=["txt", "md", "pdf", "docx"])
    if uploaded_files and st.button("💾 Save Files"):
        doc_manager.save_uploaded_files(uploaded_files)
        st.success("✅ Files have been saved!")

# VECTOR DATABASE
elif menu == "💬 Chat":
    st.markdown("## 🧠 Build a knowledge base from documents")
    col1, col2 = st.columns(2)
    if "success" not in st.session_state:
        st.session_state.success = False
    with col1:
        if st.button("⚙️ Process and Save to Chroma"):
            docs = doc_manager.load_documents("files/")
            if not docs:
                    st.warning("No available files")
            else:
                rag.save_to_chroma(docs)
                st.session_state.success = True
                st.success("✅ Database has been created!")

                
    with col2:
        if st.button("🔍 Preview Database") and st.session_state.success:
            docs, preview = rag.inspect_chroma()
            st.markdown(f"📦 Number of chunks: **{len(docs)}**")

            with st.expander("📄 Database Contents"):
                for i, chunk in enumerate(preview):
                    st.markdown(f"#### Chunk {i+1}")
                    st.code(chunk, language="markdown")
        st.markdown("## 💬 Ask questions based on the documents")

        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        user_input = st.chat_input("Ask a question...")

        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                try:
                    response = rag.query_answer(user_input)
                    if response:
                        st.markdown(response)
                        st.session_state.chat_history.append(("assistant", response))
                    else:
                        st.markdown("No answer found.")
                        st.session_state.chat_history.append(("assistant", "No answer found."))
                except FileNotFoundError:
                        msg = "The knowledge base has not been created yet. Go to the 'Build Knowledge Base' tab."
                        st.markdown(msg)
                        st.session_state.chat_history.append(("assistant", msg))
                except Exception as e:
                        msg = f"An error occurred: {e}"
                        st.error(msg)
                        st.session_state.chat_history.append(("assistant", msg))

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []

# SUMMARIZATION
elif menu == "📝 Summarize":
    st.markdown("## ✍️ Summarize a document or text")

    option = st.selectbox("Select source:", ["File", "Text"])
    text = ""

    if option == "Text":
        text = st.text_area("Paste the text to summarize", height=300)

    elif option == "File":
        files = doc_manager.show_available_files()
        if not files:
            st.warning("No available files")
        else:
            chosen_file = st.selectbox("Choose a document:", files)
            if chosen_file:
                try:
                    text = doc_manager.load_single_file_as_text(chosen_file)
                    if text is None:
                        st.warning(f"Can't load file: {chosen_file}.")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    if text.strip():
        length = st.selectbox("Length:", ["short", "medium", "long"])
        style = st.selectbox("Style:", ["informative", "technical", "simple language"])

        if st.button("✍️ Summarize"):
            with st.spinner("Generating summary..."):
                try:
                    summary = summarizer.create_summary(text, length, style)
                    st.success("✅ Summary generated!")
                    st.markdown("📄 **Summary:**")
                    st.write(summary)
                except ValueError as ve:
                    st.warning(str(ve))
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
