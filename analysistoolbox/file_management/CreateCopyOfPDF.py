# Load packages

# Declare function
def CreateCopyOfPDF(input_file, 
                    output_file, 
                    start_page=None, 
                    end_page=None):
    """
    Copy PDF file or extract specific page range to a new PDF document.

    This function creates a duplicate of a PDF file or extracts a specified page range using
    the PyPDF2 library. It provides flexible page selection, allowing users to copy the entire
    document, extract a single page, or create a new PDF from a continuous page range. The
    function preserves all content, formatting, fonts, images, and metadata from the source
    pages. Page numbers are 1-indexed for intuitive specification (page 1 is the first page).

    PDF copying and page extraction is essential for:
      * Splitting large PDFs into smaller, manageable documents
      * Extracting specific chapters or sections from reports
      * Creating document excerpts for sharing or review
      * Removing unwanted pages while preserving others
      * Preparing presentation handouts from slide decks
      * Archiving specific portions of lengthy documents
      * Creating customized document compilations
      * Reducing file sizes by extracting relevant content only

    The function uses PyPDF2's reader and writer classes to efficiently process PDF pages
    without re-rendering or quality loss. It validates page ranges to prevent errors and
    uses 1-based indexing (matching PDF viewer conventions) for user-friendly page specification.
    The output PDF maintains all original formatting, hyperlinks, and embedded content from
    the selected pages.

    Parameters
    ----------
    input_file
        Absolute or relative path to the source PDF file to copy or extract from. The file
        must exist and be a valid PDF document.
    output_file
        Path where the output PDF file will be created. Include the desired filename with
        .pdf extension. Parent directories will not be created automatically.
    start_page
        First page number to include in the output (1-indexed). If None, starts from the
        first page of the document. Defaults to None.
    end_page
        Last page number to include in the output (1-indexed, inclusive). If None, includes
        through the last page of the document. Defaults to None.

    Returns
    -------
    None
        This function does not return a value. It creates a new PDF file at the specified
        output path containing the selected pages.

    Raises
    ------
    ValueError
        If start_page is greater than end_page, indicating an invalid page range.
    FileNotFoundError
        If the input PDF file does not exist at the specified path.
    PermissionError
        If the function lacks read access to the input file or write access to the output
        location.

    Notes
    -----
    * **Page Indexing**: Uses 1-based indexing (page 1 is the first page) for intuitive use
    * **Inclusive Range**: The end_page is included in the output
    * **Content Preservation**: Maintains all formatting, fonts, images, and hyperlinks
    * **No Directory Creation**: Output directory must exist; function won't create it
    * **Overwrite Behavior**: Existing files at output_file path will be overwritten

    Examples
    --------
    # Copy entire PDF to a new file
    CreateCopyOfPDF(
        input_file='original_report.pdf',
        output_file='backup_report.pdf'
    )
    # Creates complete duplicate of the PDF

    # Extract specific page range
    CreateCopyOfPDF(
        input_file='full_document.pdf',
        output_file='chapter_3.pdf',
        start_page=25,
        end_page=45
    )
    # Extracts pages 25-45 (inclusive) to new PDF

    # Extract single page
    CreateCopyOfPDF(
        input_file='presentation.pdf',
        output_file='title_slide.pdf',
        start_page=1,
        end_page=1
    )
    # Extracts only the first page

    # Extract from specific page to end
    CreateCopyOfPDF(
        input_file='book.pdf',
        output_file='appendix.pdf',
        start_page=200
    )
    # Extracts from page 200 to the last page

    # Batch extraction workflow
    import os
    
    # Split a large PDF into chapters
    chapters = [
        (1, 50, 'chapter_1.pdf'),
        (51, 100, 'chapter_2.pdf'),
        (101, 150, 'chapter_3.pdf')
    ]
    
    for start, end, output in chapters:
        CreateCopyOfPDF(
            input_file='complete_book.pdf',
            output_file=f'chapters/{output}',
            start_page=start,
            end_page=end
        )
    # Creates separate PDF for each chapter

    """
    # Lazy load uncommon packages
    from PyPDF2 import PdfReader, PdfWriter
    
    # If start_page is None, set to 0
    if start_page is None:
        start_page = 1
    
    # If end_page is None, set to last page of PDF
    if end_page is None:
        end_page = PdfFileReader(input_file).getNumPages()
    
    # If start_page is greater than end_page, raise error
    if start_page > end_page:
        raise ValueError('start_page must be less than or equal to end_page')
    
    # Read in the input PDF
    with open(input_file, 'rb') as input_pdf:
        pdf_reader = PdfReader(input_pdf)
        pdf_writer = PdfWriter()
        
        # Add the correct pages to the output PDF
        for page in range(start_page, end_page + 1):
            pdf_writer.add_page(pdf_reader.pages[page - 1])
        
        # Write the output PDF to a file
        with open(output_file, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)

