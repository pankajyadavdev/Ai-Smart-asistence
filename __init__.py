from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF.
    """

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text