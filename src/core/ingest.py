import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS

def ingest(pdf_dir="data/pdfs", index_path="data/faiss_idx"):
    docs = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(os.path.join(pdf_dir, fname))
            docs.extend(loader.load())
    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(index_path)
        print(f"✅ Index saved at {index_path}")
        return vs
    else:
        print(f"No pdf files found in {pdf_dir}")
        return None

if __name__ == "__main__":
    ingest()