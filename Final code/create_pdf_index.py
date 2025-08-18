# create_pdf_index.py
from context_retriever import build_pdf_index

if __name__ == "__main__":
    
    ok = build_pdf_index(pdf_folder="all_pdfs", index_path="pdf_index.pkl")
    if ok:
        print("PDF index built successfully.")
    else:
        print("PDF index build failed or no PDFs found.")
