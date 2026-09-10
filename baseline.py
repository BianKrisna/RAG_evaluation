from langchain_community.document_loaders import directory, PyMuPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

loader = directory.DirectoryLoader(path="./src",
                                    loader_cls=PyMuPDFLoader)
loaded_files = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024,
    chunk_overlap = 100
)
splitted_files = text_splitter.split_documents(loaded_files)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                   encode_kwargs={'normalize_embeddings': True})
vectorstore = FAISS.from_documents(splitted_files, embeddings)
vectorstore.save_local("my_vector_db")

huggingface_token = os.getenv("HF_API")
if not huggingface_token:
    raise RuntimeError(
        "Hugging Face token is missing. Set HF_TOKEN before running the script."
    )

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
    huggingfacehub_api_token=huggingface_token,
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.1,
    task="text-generation",
    max_new_tokens=512
    )
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer concise.\n\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "{context}"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

question = input("What is your question?")
response = rag_chain.invoke({"input": question})
print(response["answer"])