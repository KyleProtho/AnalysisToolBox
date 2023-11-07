# Load packages
from PyPDF2 import PdfReader, PdfWriter

# Declare function
def CreateCopyOfPDF(input_file, 
                    output_file, 
                    start_page=None, 
                    end_page=None):
    """
    This function creates a copy of a PDF file.

    Args:
    input_file (str): The path to the input PDF file to copy (including the file name).
    output_file (str): The path to the output PDF file to create (including the file name).
    start_page (int, optional): The page number to start copying from. Defaults to None (first page).
    end_page (int, optional): The page number to stop copying at. Defaults to None (last page).

    Raises:
    None
    """
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

