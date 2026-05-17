# 1. PDF Upload + Loader
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "Python Crash Course.pdf"   # PDF file path
loader = PyPDFLoader(pdf_path)
docs = loader.load()
#print("PDF Loaded")

# 2. Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
#print(f"{len(chunks)} chunks created")

# 3. Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#print("Embeddings model loaded")

# 4. Vector Database (FAISS)
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
#print("Vector DB created")

# 5. Prompt Template
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the following context to answer:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# 6. LLM (GPT4All)
from gpt4all import GPT4All

llm = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")
#print("LLM Loaded")

# 7. Manual RAG (Retriever + LLM, no chains)
query = input("Ask your question: ")
docs = retriever.invoke(query)

# Combine context
context = "\n".join([doc.page_content for doc in docs])

# Build final prompt
final_prompt = prompt_template.format(context=context, question=query)

# Get answer from LLM
answer = llm.generate(final_prompt)

# 8. Print Answer
#print("\nFinal Answer:\n", answer)

# 9. Streaming Output (Typing Effect)
import time
print("\nStreaming Answer:\n")
for char in answer:
    print(char, end="", flush=True)
    time.sleep(0.05)
