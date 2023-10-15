# Load packages
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Declare functions
def SendPromptToChatGPT(user_prompt,
                        system_message="You are a helpful assistant.",
                        # LLM parameters
                        openai_api_key=None,
                        temperature=0.0,
                        chat_model_name="gpt-3.5-turbo",
                        verbose=True):
    # If OpenAI API key is not provided, then try to load from .env file
    if openai_api_key is None:
        load_dotenv()
        try:
            openai_api_key = os.environ['OPENAI_API_KEY']
        except:
            raise ValueError("No API key provided and no .env file found. If you need a OpenAI API key, visit https://platform.openai.com/")
    
    # Create an instance of the ChatGPT chat model
    chat_model = ChatOpenAI(openai_api_key=openai_api_key)

    # Define the system and user messages
    system_msg = SystemMessage(content=system_message)
    user_msg = HumanMessage(content=user_prompt)

    # Send the system and user messages as a one-time prompt to the chat model
    response = chat_model([system_msg, user_msg])
    
    # Print the response
    if verbose:
        print("System message:")
        print(system_message, "\n")
        print("User prompt:")
        print(user_prompt, "\n")
        print("Response:")
        print(response.content)
    
    # Return the response
    return response

 
# # Test function
# response = SendPromptToChatGPT(
#     user_prompt="""
#         Break this key intelligence question into less than four sub-questions: "Which targets are Hamas most likely to strike in their war against Israel?"
#     """
#     ,
#     system_message="""
#         You are a project manager. You specialize in taking a key intelligence question and breaking it down into sub-questions. 
#         When creating the sub-questions, identify the main components of the original question. What are the essential elements or variables that the decision maker is concerned about?
#     """,
#     openai_api_key=open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# )
