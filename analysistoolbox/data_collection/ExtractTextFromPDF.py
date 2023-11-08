# Load packages
import PyPDF2
import re

# Delclare function
def ExtractTextFromPDF(filepath_to_pdf,
                       filepath_for_exported_text,
                       start_page=1,
                       end_page=None,
                       show_word_count=False,
                       show_estimated_token_count=False):
    """
    This function extracts text from a PDF file, cleans it, then saves it to a text file.

    Args:
        filepath_to_pdf (str): The path to the PDF file that you want to extract text from.
        filepath_for_exported_text (str): The path to the text file that you want to save the extracted text to.
        start_page (int, optional): The page number of the PDF file that you want to start extracting text from. Defaults to 1.
        end_page (int or None, optional): The page number of the PDF file that you want to stop extracting text from. If None, the function will extract text from all pages in the PDF file. Defaults to None.
        show_word_count (bool, optional): Whether or not to show the word count of the extracted text. Defaults to True.
    """
    
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
