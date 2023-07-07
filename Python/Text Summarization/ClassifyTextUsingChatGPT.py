import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def ClassifyTextUsingChatGPT(text_to_classify,
                             categories,
                             openai_api_key,
                             my_prompt_template="""You will be provided with text delimited with triple backticks \
                                Classify each text into a category. \
                                Provide your output in json format with the keys: "text" and "category". \
                                Categories: {categories}. \
                                text: ```{text_to_classify}```""",
                             print_api_cost=True,
                             temperature=0.0):  
    # Set the chat prompt template
    prompt_template = ChatPromptTemplate.from_template(my_prompt_template)
    
    # Set the messages
    messages = prompt_template.format_messages(
        text_to_classify=text_to_classify,
        categories=categories
    )
      
    # Estimate the number of tokens in the prompt
    word_count = len(my_prompt_template.split())
    word_count = word_count + len(user_message.split())
    estimated_tokens = word_count * 1.33
    
    # If the estimated number of tokens is greater than 2000, use the 16k model
    if estimated_tokens > 2700:
        gpt_model = "gpt-3.5-turbo-16k"
        cost_per_1k_tokens = 0.003
    else:
        gpt_model = "gpt-3.5-turbo"
        cost_per_1k_tokens = 0.0015
        
    # Print the cost of the API usage, format as USD
    if print_api_cost:
        cost = estimated_tokens/1000 * cost_per_1k_tokens
        if cost < 0.01:
            print("Cost of API call: <$0.01")
        else:
            cost = "${:,.2f}".format(cost)
            print("Cost of API call:", cost)
    
    # Set the model name and temperature
    chatgpt_model = ChatOpenAI(
        temperature=0.0, 
        model_name=gpt_model,
        openai_api_key=openai_api_key
    )
    
    # Send the prompt to the OpenAI API 
    response = chatgpt_model(messages)
    
    # Return the response
    return(response.content)


# # Test the function
# # Read in OpenAI API key
# my_openai_api_key = open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# # Set the categories
# possible_categories = "Billing, Technical Support, \
# Account Management, or General Inquiry."
# # Set the user message
# user_message = f"""
# I want you to delete my profile and all of my user data
# """
# # Classify the text
# response = ClassifyTextUsingChatGPT(
#     text_to_classify=user_message,
#     categories=possible_categories,
#     openai_api_key=my_openai_api_key,
#     print_api_cost=True,
#     temperature=0.0
# )
# print(response)
