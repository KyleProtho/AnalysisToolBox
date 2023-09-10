# Load packages
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion

# Declare function
def CreateContextForKernel(kernel,
                           dict_context):
    # Create a context object
    my_context = kernel.create_new_context()
    
    # Iterate through context dictionary and add to context object
    for key, value in dict_context.items():
        my_context[key] = value
    
    # Return the context object
    return(my_context)


# # Test function
# strengths = [ "Unique garlic pizza recipe that wins top awards","Owner trained in Sicily at some of the best pizzerias","Strong local reputation","Prime location on university campus" ]
# weaknesses = [ "High staff turnover","Floods in the area damaged the seating areas that are in need of repair","Absence of popular calzones from menu","Negative reviews from younger demographic for lack of hip ingredients" ]
# my_context = CreateContextForKernel(
#     kernel=my_kernel,
#     dict_context={
#         "input": "makes pizzas",
#         "strengths": ", ".join(strengths),
#         "weaknesses": ", ".join(weaknesses)
#     }
# )

