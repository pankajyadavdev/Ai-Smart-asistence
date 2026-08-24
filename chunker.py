def split_pages_into_chunks(
    pages,
    chunk_size=1500,
    chunk_overlap=200
):
    """
    Split PDF pages into smaller chunks
    while keeping page numbers.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "page": page_number,
                        "text": chunk_text
                    }
                )

            start += (
                chunk_size - chunk_overlap
            )

    return chunks