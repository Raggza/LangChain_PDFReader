from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

# Cau hinh
model_file = (
    "G://Python/Flask/Flask_LangChain/PDF_reader/models/vinallama-7b-chat_q5_0.gguf"
)
vector_db_path = "G://Python/Flask/Flask_LangChain/PDF_reader/vectorstores/db_faiss"
pdf_data_path = "G://Python/Flask/Flask_LangChain/PDF_reader/data"


# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file, model_type="llama", max_new_tokens=1024, temperature=0.01
    )
    return llm


# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt


# Tao simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return llm_chain


# Read tu VectorDB
def read_vectors_db():
    # Embeding
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {"allow_download": "True"}
    embedding_model = GPT4AllEmbeddings(
        model_name=model_name, gpt4all_kwargs=gpt4all_kwargs
    )
    db = FAISS.load_local(
        vector_db_path, embedding_model, allow_dangerous_deserialization=True
    )
    return db


# Bat dau thu nghiem
db = read_vectors_db()
llm = load_llm(model_file)

# Tao Prompt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Chỉ tập trung vào câu trả lời chính. 
                Không thêm thông tin khác tuy nhiên cần tạo ra câu trả lời như đang trò chuyện giữa người với người\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)

llm_chain = create_qa_chain(prompt, llm, db)

# Chay cai chain
question = "Bài viết đề cập đến vấn đề gì"
response = llm_chain.invoke({"query": question})
result = response.get("result", "")
print(result)
