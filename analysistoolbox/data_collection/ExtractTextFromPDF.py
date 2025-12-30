# Load packages
import re

# Delclare function
def ExtractTextFromPDF(filepath_to_pdf,
                       filepath_for_exported_text,
                       start_page=1,
                       end_page=None,
                       show_word_count=False,
                       show_estimated_token_count=False):
    """
    Extract text from a PDF file, clean it, and save to a text file.

    This function reads a PDF document page by page, extracts the text content,
    applies basic cleaning operations (removing extra whitespace, handling hyphenation),
    and writes the cleaned text to a specified output file. Optionally, it can display
    word count and estimated token count statistics for the extracted content.

    Text cleaning operations include:
      * Stripping leading and trailing whitespace
      * Removing duplicate spaces
      * Handling hyphenation at line breaks
      * Removing line breaks between continuous words

    This is particularly useful for preparing PDF content for further text analysis,
    machine learning applications, or creating clean text corpora from document collections.

    Parameters
    ----------
    filepath_to_pdf
        Path to the PDF file to extract text from. Must be a string ending in '.pdf'.
    filepath_for_exported_text
        Path where the extracted text will be saved. Must be a string ending in '.txt'.
    start_page
        Page number to start extracting text from (1-indexed). Defaults to 1.
    end_page
        Page number to stop extracting text from. If None, extracts text from all pages
        in the PDF file. Defaults to None.
    show_word_count
        If True, displays the word count of the extracted text. Defaults to False.
    show_estimated_token_count
        If True, displays an estimated token count based on word count. Defaults to False.

    Returns
    -------
    None
        This function writes to a file and optionally prints statistics, but does not
        return a value.

    Examples
    --------
    # Extract all text from a PDF and save to a text file
    ExtractTextFromPDF(
        filepath_to_pdf='document.pdf',
        filepath_for_exported_text='output.txt',
        show_word_count=True
    )

    # Extract text from pages 5 through 10 only
    ExtractTextFromPDF(
        filepath_to_pdf='document.pdf',
        filepath_for_exported_text='output.txt',
        start_page=5,
        end_page=10
    )

    """
    # Lazy load uncommon packages
    import PyPDF2
    
    # Ensure that filepath_to_pdf is a string ending in .pdf
    if not isinstance(filepath_to_pdf, str) or not filepath_to_pdf.endswith('.pdf'):
        raise TypeError("filepath_to_pdf must be a string ending in .pdf")  
    
    # Ensure that filepath_for_exported_text is a string ending in .txt
    if not isinstance(filepath_for_exported_text, str) or not filepath_for_exported_text.endswith('.txt'):
        raise TypeError("filepath_for_exported_text must be a string ending in .txt")
    
    # Open the PDF file in read-binary mode
    with open(filepath_to_pdf, 'rb') as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # If end_page is not specified, set it to the number of pages in the PDF
        if end_page == None:
            # Get the number of pages in the PDF document
            num_pages = len(pdf_reader.pages)
            end_page = num_pages

        # Create a new text file to write the extracted text to
        with open(filepath_for_exported_text, 'w', encoding='utf-8') as text_file:
            # Loop through each page in the PDF document
            for page_num in range(start_page-1, end_page):
                # Get the page object for the current page
                page_obj = pdf_reader.pages[page_num]

                # Extract the text from the current page
                page_text = page_obj.extract_text()
                
                # Clean up the text
                page_text = page_text.strip()
                page_text = page_text.replace("  ", " ")
                page_text = page_text.replace(" -", "-")
                
                # Use regex to replace all instances of a space and new line between to letters
                page_text = re.sub(r'(?<=[a-zA-Z]) \n(?=[a-zA-Z])', '', page_text)
                
                # Write the text from the current page to the text file
                text_file.write(page_text)

    # Print a message to indicate that the text has been extracted and saved
    print("Text extracted and saved to " + filepath_for_exported_text + ".")
    
    # Read the text file if word count or estimated token count is True
    if show_word_count or show_estimated_token_count:
        final_text = open(filepath_for_exported_text, 'r', encoding='utf-8').read()
        
    # Show the word count if show_word_count is True
    if show_word_count:
        # Split the text into a list of words
        final_text = final_text.split()
        # Get the number of words in the text
        word_count = len(final_text)
        # Print the number of words in the text
        print(f"Word count: {word_count}")
        
    # Show the estimated token count if show_estimated_token_count is True
    if show_estimated_token_count:
        # Get the number of words in the text
        word_count = len(final_text)
        # Get the number of tokens in the text
        number_of_tokens = word_count * (100/75)
        print(f"Estimated token count: {int(round(number_of_tokens, 0))}")
