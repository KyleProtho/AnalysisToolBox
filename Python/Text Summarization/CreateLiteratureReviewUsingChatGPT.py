import openai
import PyPDF2
import re
import tempfile

def CreateLiteratureReviewUsingChatGPT(filepath_to_pdf,
                                       openai_api_key,
                                       start_page=1,
                                       end_page=None,
                                       print_api_cost=True,
                                       temperature=0.20):
    """This function sends a prompt to the OpenAI API and returns the response. It also prints the cost of the API call.

    Args:
        prompt (str): The prompt to send to the OpenAI API.
        openai_api_key (str): The OpenAI API key.
        print_api_cost (bool, optional): Whether to print the estimated cost of the API call. Defaults to True.
        temperature (float, optional): The temperature of the response. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 0.4.
    """
    
    # Ensure that filepath_to_pdf is a string ending in .pdf
    if not isinstance(filepath_to_pdf, str) or not filepath_to_pdf.endswith('.pdf'):
        raise TypeError("filepath_to_pdf must be a string ending in .pdf")  
    
    # Open the PDF file in read-binary mode
    with open(filepath_to_pdf, 'rb') as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # If end_page is not specified, set it to the number of pages in the PDF
        if end_page == None:
            # Get the number of pages in the PDF document
            num_pages = len(pdf_reader.pages)
            end_page = num_pages
            
        # Create a temp directory to store the extracted text
        temp_dir = tempfile.TemporaryDirectory()
        filepath_for_exported_text = temp_dir.name + "/extracted_text.txt"

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
    
    # Read in the extracted text, then delete the temp directory
    extracted_text = open(filepath_for_exported_text, 'r', encoding='utf-8').read()
    temp_dir.cleanup()
    
    # Set the prompt
    prompt = f"""
    Conduct a literature review of this academic paper, and write your response at an 8th grade reading level. 
    
    Be sure to include the following:
    - The title of the paper
    - The methodology used in the paper
    - The results of the paper
    - Potential future research ideas
    
    Format your response as a JSON object with the following keys:
    - title
    - methodology
    - results
    - future_research
    
    Text from academic paper:
    {extracted_text}
    """
    
    # Estimate the number of tokens in the prompt
    word_count = len(prompt.split())
    word_count = word_count + (len(extracted_text) / 4.7)  # Avg. length of word is 4.7 characters
    estimated_tokens = word_count * 1.33
    print("Estimated number of tokens:", estimated_tokens)
    
    # If the estimated number of tokens is greater than 2000, use the 16k model
    if estimated_tokens > 2700:
        gpt_model = "gpt-3.5-turbo-16k"
        cost_per_1k_tokens = 0.003
    else:
        gpt_model = "gpt-3.5-turbo"
        cost_per_1k_tokens = 0.0015
        
    # Set the OpenAI API key
    openai.api_key = openai_api_key
    
    # Send the prompt to the OpenAI API 
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature
    )

    # Print the cost of the API usage, format as USD
    if print_api_cost:
        cost = response['usage']['total_tokens']/1000 * cost_per_1k_tokens
        if cost < 0.01:
            print("Cost of API call: <$0.01")
        else:
            cost = "${:,.2f}".format(cost)
            print("Cost of API call:", cost)
            
    # Extract the content from the response
    content = response['choices'][0]['message']['content']
    
    # Return the content
    return(content)


# # Test the function
# # Get OpenAI API key
# my_openai_api_key = open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# # # Create literature review
# # literature_review_json = CreateLiteratureReviewUsingChatGPT(
# #     filepath_to_pdf="C:/Users/oneno/Downloads/TheRiskofUsingRiskMatrices.pdf", 
# #     openai_api_key=my_openai_api_key,
# #     start_page=2,
# #     end_page=3
# # )
# # # Print the literature review JSON in a readable format
# # print(literature_review_json)
# # Create literature review
# literature_review_json = CreateLiteratureReviewUsingChatGPT(
#     filepath_to_pdf="C:/Users/oneno/Downloads/TheRiskofUsingRiskMatrices.pdf", 
#     openai_api_key=my_openai_api_key,
#     start_page=2,
#     end_page=5
# )
# # Print the literature review JSON in a readable format
# print(literature_review_json)
