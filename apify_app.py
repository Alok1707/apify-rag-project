import os
from dotenv import load_dotenv

from langchain_apify import ApifyWrapper
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load secret keys from the .env file into Python's memory (os.environ)
load_dotenv()

# 2. Initialize the Apify web crawler and the Google Gemini LLM
apify = ApifyWrapper()
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

# 3. Set up local CPU embeddings to convert text chunks into numbers (100% free & local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Calling website content crawler...")
# 4. Crawl the target website and return the loader object
raw_document = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={
        "startUrls": [{"url": "https://docs.langchain.com/oss/python/langchain/quickstart"}],
        "maxCrawlPages": 10,
        "crawlerType": "cheerio"
    },
    dataset_mapping_function=lambda item: Document(
        page_content=item.get("text") or "", 
        metadata={"source": item.get("url", "")}
    ),
)

print("Splitting documents into optimal chunks...")
# 5. Initialize the text splitter and fetch documents using .load() before splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_documents(raw_document.load())

print(f"Created {len(docs)} text chunks. Storing in vector database...")
# 6. Store the chunked documents in the temporary local vector database
vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
retriever = vector_store.as_retriever()

# 7. Create a prompt template instructing Gemini how to answer using only the provided context
prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

# 8. Ask a question, search for relevant text chunks, and generate an answer
query = "What is LangChain?"
retrieved_docs = retriever.invoke(query)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

# Combine the prompt and LLM into a chain, pass variables, and extract the text content
answer = (prompt | llm).invoke({"context": context, "question": query}).content
sources = ", ".join({doc.metadata.get("source", "") for doc in retrieved_docs if doc.metadata.get("source")})

print("\nAnswer:", answer)
print("Source:", sources)