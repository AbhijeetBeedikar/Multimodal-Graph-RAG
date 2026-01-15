from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PDFMinerLoader
from langchain_core.documents import Document


import os
# def text_cleaning():
  #NEEDS TO BE DONE BEFORE TEXT SPLITTING (If time permits)
def text_split(path):
  print(os.getcwd())
  if path.lower().endswith(".pdf"):
    loader = PDFMinerLoader(path)
    docs = loader.load()
  else:
    with open(path, 'r') as f:
      pageContent = f.read()
    docs = Document(
      page_content=pageContent,
      metadata={
        "source_file": path,
        "modality": "text",
        "chunk_index": 0,
      }
    )
    
  if len(docs) == 0:
    raise ValueError("PDF text extraction failed or returned empty document.")
  #docs.page_content = text_cleaning(docs.page_content)

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,  # chunk size (characters)
      chunk_overlap=200,  # chunk overlap (characters) Starting 200 chars are the same as prev 200 chars to maintain context
      add_start_index=True,  # track index in original document
  )
  all_splits = text_splitter.split_documents(docs)

  
  print(f"Split blog post into {len(all_splits)} sub-documents.")
  return [Document(page_content=all_splits[i].page_content,metadata={**all_splits[i].metadata,"modality":"pdf","chunk_index":i}) for i in range(len(all_splits))]
