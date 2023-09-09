# Load packages
import semantic_kernel as sk

# Declare function
def CreateSemanticFunction(kernel,
                           prompt_template,
                           function_description=None,
                           max_tokens=2000,
                           temperature=0.1,
                           top_p=0.5):
    # Create the semantic function
    semantic_function = kernel.create_semantic_function(
        prompt_template=prompt_template,
        description=function_description,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    print("A semantic function for summarization has been registered to your kernel.")
    
    # Return the semantic function
    return(semantic_function)


# # Test the function. Remember to run the CreateSemanticKernel function first.
# SWOT_analysis_function = CreateSemanticFunction(
#     kernel=my_kernel,
#     prompt_template="""
#     {{$input}}

#     Convert the analysis provided above to the business domain of {{$domain}}.
#     """,
#     function_description="Convert SWOT analysis questions to a different business domain."
# )
# # Create a context for the template
# my_context = kernel.create_new_context()
# my_context['input'] = """
# 1. **Strengths**
#     - What unique recipes or ingredients does the pizza shop use?
#     - What are the skills and experience of the staff?
#     - Does the pizza shop have a strong reputation in the local area?
#     - Are there any unique features of the shop or its location that attract customers?
# 2. **Weaknesses**
#     - What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)
#     - Are there financial constraints that limit growth or improvements?
#     - Are there any gaps in the product offering?
#     - Are there customer complaints or negative reviews that need to be addressed?
# 3. **Opportunities**
#     - Is there potential for new products or services (e.g., catering, delivery)?
#     - Are there under-served customer segments or market areas?
#     - Can new technologies or systems enhance the business operations?
#     - Are there partnerships or local events that can be leveraged for marketing?
# 4. **Threats**
#     - Who are the major competitors and what are they offering?
#     - Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?
#     - Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?
#     - Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?
# """
# my_context['domain'] = "construction management"
# # Run the semantic function
# result = await kernel.run_async(SWOT_analysis_function, input_context=my_context)
# from IPython.display import display, Markdown
# display(Markdown(f"### ✨ Shift the SWOT interview questions to the world of {my_context['domain']}\n"+ str(result)))  
