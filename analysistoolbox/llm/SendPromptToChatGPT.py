# Load packages

# Declare functions
def SendPromptToChatGPT(prompt_template,
                        user_input,
                        system_message="You are a helpful assistant.",
                        # LLM parameters
                        openai_api_key=None,
                        temperature=0.0,
                        chat_model_name='gpt-4o-mini',
                        maximum_tokens=1000,
                        verbose=True):
    # Lazy load uncommon packages
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    import openai
    
    # If OpenAI API key is not provided, then try to load from .env file
    if openai_api_key is None:
        raise ValueError("No API key provided. If you need a OpenAI API key, visit https://platform.openai.com/")
    
    # Ensure that user_input is a dictionary
    if type(user_input) != dict:
        raise ValueError("user_input must be a dictionary with the variable in the prompt template as the key and text you want plugged into the template as the value.")
    
    # Ensure that each key in user_input is in the prompt template
    for key in user_input.keys():
        if "{" + key + "}" not in prompt_template:
            raise ValueError("The key '" + key + "' in user_input is not in the prompt template.")
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create an instance of the ChatGPT chat model
    chat_model = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=chat_model_name,
        max_tokens=maximum_tokens
    )
    
    # Create the ouput parser
    output_parser = StrOutputParser()
    
    # Create chain using lanchain's expression language (LCEL)
    chain = prompt | chat_model | output_parser

    # Send the system and user messages as a one-time prompt to the chat model
    response = chain.invoke(user_input)
    
    # Return the response
    return response

