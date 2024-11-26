from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


def pdf_to_text(pdf_dirpath,output_text_file_pat):
    papers = []
    loader = DirectoryLoader(pdf_dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
    try:
        papers = loader.load()
    except Exception as e:
        print(f"Error loading file: {e}")
    full_text = ''
    for paper in papers:
        full_text += paper.page_content

    full_text = " ".join(line for line in full_text.splitlines() if line)
    print(full_text)
    # Save the full text to a text file
    try:
        with open(output_text_file_pat, "w", encoding="utf-8") as file:
            file.write(full_text)
        print(f"Content saved to {output_text_file_pat}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__=='__main__':

    pdf_dirpath = "data"
    output_text_file_path = "outputs/paper_16.txt"
    
    pdf_to_text(pdf_dirpath,output_text_file_path)

    