# The application of Langchain in tuning a question-and-answer model for PDF content
 
## Mô tả
Ứng dụng Langchain trong việc tinh chinh model LLM sử dụng trong hỏi đáp dựa trên văn bản PDF.

## Mô hình
Mô hình LLM sử dụng trong dự án dưới đấy được cung cấp bởi Hugging Face. Tìm hiểu thêm về mô hình theo đường link dưới đây:
- [vilm/vinallama-7b-chat-GGUF](https://huggingface.co/vilm/vinallama-7b-chat-GGUF)

## Cách sử dụng:
1. [prepare_db_store.py](https://github.com/Raggza/LangChain_PDFReader/blob/main/prepare_db_store.py): Chứa hàm tạo ra vector database dùng để lưu trữ dữ liệu dạng vector.
2. [qabot.py](https://github.com/Raggza/LangChain_PDFReader/blob/main/qabot.py): Chương trình chính dùng để hỏi đáp.
3. [requirements.txt](https://github.com/Raggza/LangChain_PDFReader/blob/main/requirements.txt): Các thư viện cần dùng trong chương trình.
4. [data](https://github.com/Raggza/LangChain_PDFReader/tree/main/data): Thư mục chứa các file PDF.

**Lưu ý:**
1. Thay đổi các đường dẫn phù hợp với setup của người dùng.

## Kết quả:

![alt text](https://github.com/Raggza/hinh/blob/main/PDF_Reader/hinh1.png)
