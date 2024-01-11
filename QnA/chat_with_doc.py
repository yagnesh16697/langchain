import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma


# Load document
def load_document(file):
    import os

    name, extension = os.path.splitext(file)

    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader

        print(f"Loading {file}")
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader

        print(f"Loading {file}")
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain.document_loaders import TextLoader

        print(f"Loading {file}")
        loader = TextLoader(file)
    else:
        print("Document type not supported")
        return None

    data = loader.load()
    return data


# Chunk document
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks


# Create embeddings
def create_embeddings(chunks):
    from langchain.embeddings import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore


# QnA
def ask_question(question, vectorstore):
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    chain = load_qa_chain(llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(question=question)
    return answer


# Calculate cost
def calculate_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    total_price = total_tokens / 1000 * 0.0004

    return total_price, total_tokens


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)

    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/1*odEY2uy37q-GTb8-u7_j8Q.png",
        width=200,
    )
    st.subheader("LLM Chat with Document")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        uploaded_file = st.file_uploader(
            "Upload Document", type=["pdf", "docx", "txt"], accept_multiple_files=False
        )
        chunk_size = st.number_input(
            "Chunk Size", min_value=100, max_value=2048, value=512
        )
        k = st.number_input("k", min_value=1, max_value=20, value=3)
        add_data = st.button("Add Data")

        if uploaded_file and add_data:
            with st.spinner("Reading, Splitting and Embedding..."):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)

                st.write(f"Chunk size: {chunk_size} Chunks: {len(chunks)}")
                total_price, total_tokens = calculate_embedding_cost(chunks)
                st.write(f"Tokens: {total_tokens} Cost: ${total_price}")

                vectorstore = create_embeddings(chunks)

                st.session_state["vectorstore"] = vectorstore
                st.success("File uploaded and embeddings created")
