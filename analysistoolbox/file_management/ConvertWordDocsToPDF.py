# Load packages
import sys
import os

# Declare function
def ConvertWordDocsToPDF(word_folder_path,
                         pdf_folder_path,
                         open_each_doc=False):
    """
    Batch convert Word documents to PDF format using Microsoft Word automation.

    This function automates the conversion of all Word documents (.docx) in a specified folder
    to PDF format using Windows COM (Component Object Model) automation with Microsoft Word.
    It leverages Word's native PDF export capabilities to ensure high-fidelity conversion,
    preserving formatting, fonts, images, and layout. The function processes all .docx files
    in the source folder and saves the PDFs to a destination folder, automatically creating
    the output directory if needed.

    Batch Word-to-PDF conversion is essential for:
      * Document archival and long-term preservation in universal format
      * Creating read-only versions for distribution and sharing
      * Preparing documents for printing and professional publishing
      * Compliance and regulatory documentation requirements
      * Reducing file sizes while maintaining visual fidelity
      * Ensuring consistent viewing across different platforms and devices
      * Automating report generation and document workflows
      * Creating portfolios and document collections

    The function uses Microsoft Word's COM interface, which requires Word to be installed
    on the system (Windows only). It can operate in visible or invisible mode, allowing
    users to monitor the conversion process or run it silently in the background. The
    function automatically handles directory creation and provides progress feedback for
    each converted document.

    Parameters
    ----------
    word_folder_path
        Absolute path to the folder containing Word documents (.docx files) to convert.
        All .docx files in this folder will be processed.
    pdf_folder_path
        Absolute path to the folder where converted PDF files will be saved. The folder
        will be created automatically if it doesn't exist. PDF files will have the same
        names as the source Word documents.
    open_each_doc
        Whether to make Microsoft Word visible during conversion. If True, Word opens
        visibly for each document, allowing monitoring but slowing the process. If False,
        conversion runs silently in the background for faster processing. Defaults to False.

    Returns
    -------
    None
        This function does not return a value. It prints progress messages to stdout and
        creates PDF files in the specified output folder.

    Notes
    -----
    * **Windows Only**: Requires Microsoft Word to be installed and uses win32com library
    * **File Format**: Only processes .docx files; older .doc format not supported
    * **Overwrite Behavior**: Existing PDF files with the same name will be overwritten
    * **Error Handling**: Automatically creates output directory if it doesn't exist
    * **Performance**: Processing time depends on document complexity and count

    Examples
    --------
    # Basic batch conversion with silent processing
    ConvertWordDocsToPDF(
        word_folder_path='C:/Documents/Reports',
        pdf_folder_path='C:/Documents/Reports_PDF'
    )
    # Converts all .docx files silently, creates output folder if needed

    # Monitor conversion process with visible Word
    ConvertWordDocsToPDF(
        word_folder_path='C:/Users/John/Contracts',
        pdf_folder_path='C:/Users/John/Contracts_Archive',
        open_each_doc=True
    )
    # Shows Word window during conversion for monitoring

    # Automated report distribution workflow
    import os
    from datetime import datetime
    
    # Define paths
    source_folder = 'C:/Reports/Monthly'
    archive_folder = f'C:/Reports/Archive/{datetime.now().strftime("%Y-%m")}'
    
    # Convert all monthly reports to PDF for distribution
    ConvertWordDocsToPDF(
        word_folder_path=source_folder,
        pdf_folder_path=archive_folder,
        open_each_doc=False
    )
    # Creates monthly archive folder and converts all reports

    """
    # Lazy load uncommon packages
    import win32com.client as client
    
    # Create list of word files in the directory
    list_word_docs = [f for f in os.listdir(word_folder_path) if f.endswith(".docx")]
    
    # Connect to Microsoft Word
    word = client.Dispatch('Word.Application')
    
    # Open the word document without opening the application, unless open_each_doc is True
    if open_each_doc == False:
        word.Visible = False
    else:
        word.Visible = True
    
    # Convert each word file to PDF
    for word_doc in list_word_docs:
        # Create full path to Word document
        word_doc_path = os.path.join(word_folder_path, word_doc)
        # Create full path to PDF document
        pdf_doc_path = os.path.join(pdf_folder_path, word_doc.replace(".docx", ".pdf"))
        print("Converting {} to PDF...".format(word_doc))
        # Try to save the document as PDF
        current_doc = word.Documents.Open(word_doc_path)
        try:
            current_doc.SaveAs(pdf_doc_path, FileFormat=17)
        except Exception as e:
            # Create PDF folder if it doesn't exist
            os.makedirs(name=pdf_folder_path)
            current_doc.SaveAs(pdf_doc_path, FileFormat=17)
        current_doc.Close()
    word.Quit()
        
    # Print message saying script is done
    print("All Word documents converted to PDF.")

