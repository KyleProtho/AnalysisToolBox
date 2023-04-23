def SendPromptToChatGPT(prompt,
                        filepath_for_exported_csv,
                        openai_api_key,
                        print_api_cost=True,
                        temperature=0.4):
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
        cost = data_measure_csv['usage']['total_tokens']/1000 * 0.002
        if cost < 0.01:
            print("Cost of API call: <$0.01")
        else:
            cost = "${:,.2f}".format(cost)
            print("Cost of API call:", cost)
            
    # Extract the content from the response
    content = response['choices'][0]['message']['content']
    
    # Return the content
    return(content)
