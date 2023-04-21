import PyPDF2
import re

def ExtractTextFromPDF(filepath_to_pdf,
                       filepath_for_exported_text,
                       start_page=1,
                       end_page=None,
                       show_word_count=True,
                       show_estimated_token_count=True):
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
            num_pages = pdf_reader.getNumPages()
            end_page = num_pages

        # Create a new text file to write the extracted text to
        with open(filepath_for_exported_text, 'w', encoding='utf-8') as text_file:
            # Loop through each page in the PDF document
            for page_num in range(start_page-1, end_page+1):
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
                text_file.write(f"Page {page_num + 1}:\n{page_text}\n")

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
        # Print estimated cost of input into Chat GPT 3.5 Turbo model
        estimated_cost = round(number_of_tokens * (0.002/1000), 2)
        print(f"Estimated cost of input into Chat GPT 3.5 Turbo model: ${estimated_cost} ($0.002 per 1k tokens).")

# # Test the function
# ExtractTextFromPDF(
#     filepath_to_pdf="C:/Users/oneno/OneDrive/Creations/Star Sense/StarSense/App/data/source/2023-QRS-Measure-Technical-Specifications-Updated-October-508-Final.pdf",
#     filepath_for_exported_text="C:/Users/oneno/OneDrive/Creations/Star Sense/StarSense/App/data/source/MY 2022 Measure Specifications.txt",
#     start_page=70,
#     end_page=194
# )
