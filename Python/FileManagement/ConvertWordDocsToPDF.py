import sys
import os
import win32com.client as client

def ConvertWordDocsToPDF(word_folder_path,
                         pdf_folder_path,
                         open_each_doc=False):
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

# # Test the function
# ConvertWordDocsToPDF(
#     word_folder_path=r"C:\Users\oneno\Downloads\Test Word Docs",
#     pdf_folder_path=r"C:\Users\oneno\Downloads\Test PDF Docs"
# )
