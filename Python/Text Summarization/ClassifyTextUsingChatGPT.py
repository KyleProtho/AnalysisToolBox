import openai

def ClassifyTextUsingChatGPT(text_to_classify,
                             openai_api_key,
                             system_instruction=None,
                             delimiter="####",
                             print_api_cost=True,
                             temperature=0.0):
    if system_instruction is None:
        system_instruction = f"""
        You will be provided with text delimited with {delimiter} characters.
        Classify each query into a primary category and a secondary category. 
        Provide your output in json format with the keys: primary and secondary.
        """
        
    # Set the user message
    messages = [  
        {
            'role':'system', 
            'content': system_instruction
        },    
        {
            'role':'user', 
            'content': f"{delimiter}{user_message}{delimiter}"},  
    ] 
    
    # Set the OpenAI API key
    openai.api_key = openai_api_key
    
    # Estimate the number of tokens in the prompt
    word_count = len(system_instruction.split())
    word_count = word_count + len(user_message.split())
    estimated_tokens = word_count * 1.33
    
    # If the estimated number of tokens is greater than 2000, use the 16k model
    if estimated_tokens > 2700:
        gpt_model = "gpt-3.5-turbo-16k"
        cost_per_1k_tokens = 0.003
    else:
        gpt_model = "gpt-3.5-turbo"
        cost_per_1k_tokens = 0.0015
    
    # Send the prompt to the OpenAI API 
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
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
# # Read in OpenAI API key
# my_openai_api_key = open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# # Set the delimiter
# my_delimiter="####"
# # Set system message
# my_system_message = f"""
# You will be provided with customer service queries. \
# The customer service query will be delimited with \
# {my_delimiter} characters.
# Classify each query into a primary category \
# and a secondary category. 
# Provide your output in json format with the \
# keys: primary and secondary.

# Primary categories: Billing, Technical Support, \
# Account Management, or General Inquiry.

# Billing secondary categories:
# Unsubscribe or upgrade
# Add a payment method
# Explanation for charge
# Dispute a charge

# Technical Support secondary categories:
# General troubleshooting
# Device compatibility
# Software updates

# Account Management secondary categories:
# Password reset
# Update personal information
# Close account
# Account security

# General Inquiry secondary categories:
# Product information
# Pricing
# Feedback
# Speak to a human

# """
# # Set the user message
# user_message = f"""
# I want you to delete my profile and all of my user data
# """
# # Classify the text
# response = ClassifyTextUsingChatGPT(
#     text_to_classify=user_message,
#     openai_api_key=my_openai_api_key,
#     system_instruction=my_system_message,
#     delimiter=my_delimiter,
#     print_api_cost=True,
#     temperature=0.0
# )
# print(response)
