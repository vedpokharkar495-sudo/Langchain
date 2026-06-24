

# pip install -U langchain-community pypdf

from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_my_documents():
    # 1. Load a standard Text (.txt) file
    try:
        txt_loader = TextLoader("sample.txt", encoding="utf-8")
        txt_docs = txt_loader.load()
        print(f"--- Successfully loaded {len(txt_docs)} text document(s) ---")
        print(f"Content Sample:\n{txt_docs[0].page_content[:150]}\n")
        print(f"Metadata: {txt_docs[0].metadata}\n")
    except FileNotFoundError:
        print("Skipping text load: 'sample.txt' not found.\n")

    # 2. Load a PDF (.pdf) file
    try:
        pdf_loader = PyPDFLoader("sample.pdf")
        # .load() reads the entire PDF; each page becomes a separate Document object
        pdf_docs = pdf_loader.load()
        print(f"--- Successfully loaded {len(pdf_docs)} PDF page(s) ---")
        print(f"Page 1 Content Sample:\n{pdf_docs[0].page_content[:150]}\n")
        print(f"Page 1 Metadata: {pdf_docs[0].metadata}\n")
    except FileNotFoundError:
        print("Skipping PDF load: 'sample.pdf' not found.\n")

if __name__ == "__main__":
    # Note: Ensure 'sample.txt' or 'sample.pdf' exist in your folder to see them run!
    load_my_documents()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 1. The Document Standard Structure:No matter what file type you ingest, LangChain converts it into a uniform Document object. This object always has two main properties:
    
                 # A. page_content: A string containing the actual text extracted from the file.

               # B. metadata: A Python dictionary storing tracking data about the source (e.g., {'source': 'sample.pdf', 'page': 0}).


# 2. TextLoader:This is the simplest loader. It reads an entire raw text file and packs all of its contents into a single Document object. Specifying encoding="utf-8" prevents crashes if your file contains special characters or emojis.


# 3. PyPDFLoader:PDFs are more complex than text files because they have multiple pages. This loader automatically parses the PDF page-by-page. If your PDF has 5 pages, .load() outputs a list containing 5 distinct Document objects. It automatically populates the metadata with the correct page numbers.



# 4. The .load() Method:Every document loader in LangChain features a synchronous .load() function. This triggers the actual disk-read operation and returns a Python list of documents, making it easy to pass directly into text splitters or vector databases later.



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

