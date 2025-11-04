from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
# loader = TextLoader("demo1.txt", encoding="utf-8")
# docs = loader.load()
# print(docs[0])
loader = DirectoryLoader(
    "docs/",
    glob="**/*.txt",
    loader_cls = lambda path: TextLoader(path, encoding="utf-8")
)
docs = loader.load()
print("Total docs:", len(docs))
# for d in docs:
#     print("CONTENT:\n", d.page_content[:300])
#     print("METADATA:", d.metadata)
#     print("----")
load_dotenv()
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)
print("Chunks created: ", len(chunks))
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("INDEX_NAME")
# Create index if not exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(INDEX_NAME)
# pinecone
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embed_model,
    index_name=INDEX_NAME
)
print("Embedding stored in pinecone")
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs={"k": 4}
)
# query = "What is the investigation about?"
# results = retriever.invoke(query)
# for r in results:
#     print(r.page_content[:300], "\n---")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key = os.getenv("API_KEY"),
    temperature = 0.2
)
prompt = PromptTemplate(
    template="""
You are an expert assistant. Use ONLY the following context to answer.
If the answer is not in the context, say:
"I could not find the answer in the document."
Context:
{context}
Question:
{question}
Answer:
""",
    input_variables=["context", "question"]
)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
query = "When is the investigation reveals?"
print("\nAsking:", query)
answer = rag_chain.invoke(query)
print("\nFinal Answer:\n")
print(answer)







