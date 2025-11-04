from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

import chromadb
import os
from dotenv import load_dotenv

load_dotenv()


def load_documents(path: str = None, urls: list = None):
    docs = []
    if path and os.path.exists(path):
        pdf_loader = DirectoryLoader(path, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
        docs.extend(pdf_loader.load())

        text_loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
        docs.extend(text_loader.load())
    elif urls:
        for url in urls:
            docs.extend(WebBaseLoader(url).load())
    else:
        raise ValueError("Provide either a valid local path or URLs list.")
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(path: str):
    docs = load_documents(path=path)
    print(f"✅ Loaded {len(docs)} documents")

    chunks = split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    
    collection_name="Langchain_store"
    
    # 🔥 [IMPORTANT] Drop the old collection if it exists
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        print(f"⚠️ Collection '{collection_name}' already exists. Deleting old data...")
        client.delete_collection(collection_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="Langchain_store",
        client=client,
    )

    print("✅ Vector store on Chroma Cloud built / updated")
    return vectorstore


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("API_KEY"),
        temperature=0.2
    )
    
    # 🧠 Modern Memory — InMemoryChatMessageHistory
    chat_history = InMemoryChatMessageHistory()


    prompt = PromptTemplate(
        template=(
            "You are an expert assistant. Use ONLY the following context to answer.\n"
            "If the answer is not in the context, say: "
            "\"I could not find the answer in the document.\"\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )
    rag_chain = (
        {
            # :white_check_mark: Only pass the string question to the retriever
            "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
            "question": RunnablePassthrough(),
             "chat_history": RunnableLambda(lambda x: "\n".join(
                [f"Human: {m.content}" if isinstance(m, HumanMessage)
                 else f"AI: {m.content}" for m in chat_history.messages]
             )),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, chat_history


if __name__ == "__main__":
    path = r"E:\P99Trainee\Python\AI\Langchain\Test_Docs"

    vector_store = build_vector_store(path)
    rag_chain, chat_history = build_rag_chain(vector_store)

    print("💬 Chat with your document! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = rag_chain.invoke({"question": query})
        print("\n🤖 AI:", result)

        # 🧠 Update memory
        chat_history.add_message(HumanMessage(content=query))
        chat_history.add_message(AIMessage(content=result))

        # [CHECK] View stored messages
        print("\n📜 Chat memory so far:")
        for msg in chat_history.messages:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"{role}: {msg.content}")
        print("-" * 50)
