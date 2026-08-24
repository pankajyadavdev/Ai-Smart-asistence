from pypdf import PdfReader


def extract_pages_from_pdf(pdf_file):
    """
    Extract text from each PDF page while
    preserving the page number.
    """

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    return pages