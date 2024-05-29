from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

vector_db_path = "G://Python/Flask/Flask_LangChain/PDF_reader/vectorstores/db_faiss"
pdf_data_path = "G://Python/Flask/Flask_LangChain/PDF_reader/data"


def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {"allow_download": "True"}
    embedding_model = GPT4AllEmbeddings(
        model_name=model_name, gpt4all_kwargs=gpt4all_kwargs
    )
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db


create_db_from_files()
