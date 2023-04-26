import openai

def SendPromptToChatGPT(prompt,
                        openai_api_key,
                        print_api_cost=True,
                        temperature=0.4):
    """This function sends a prompt to the OpenAI API and returns the response. It also prints the cost of the API call.

    Args:
        prompt (str): The prompt to send to the OpenAI API.
        openai_api_key (str): The OpenAI API key.
        print_api_cost (bool, optional): Whether to print the estimated cost of the API call. Defaults to True.
        temperature (float, optional): The temperature of the response. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 0.4.
    """
    
    # Set the OpenAI API key
    openai.api_key = openai_api_key
    
    # Send the prompt to the OpenAI API 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature
    )

    # Print the cost of the API usage, format as USD
    if print_api_cost:
        cost = response['usage']['total_tokens']/1000 * 0.002
        if cost < 0.01:
            print("Cost of API call: <$0.01")
        else:
            cost = "${:,.2f}".format(cost)
            print("Cost of API call:", cost)
            
    # Extract the content from the response
    content = response['choices'][0]['message']['content']
    
    # Return the content
    return(content)

