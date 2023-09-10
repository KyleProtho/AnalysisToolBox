# Load packages
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion

# Declare function
def CreateSemanticKernel(kernel_platform="openai",
                         openai_api_key=None,
                         openai_org_id=None,
                         platform_model_name="gpt-3.5-turbo"):
    # Create a kernel
    kernel = sk.Kernel()
    
    # Set the kernel platform. Setup Azure OpenAI kernel
    if kernel_platform == "azureopenai":
        deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
        kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
        print("You made a kernel using GPT model on Azure OpenAI")
    else:
        useAzureOpenAI = False
        
        # Setup OpenAI kernel
        if kernel_platform == "openai":
            if openai_org_id is None and openai_api_key is None:
                openai_api_key, openai_org_id = sk.openai_settings_from_dot_env()
            kernel.add_text_completion_service("openai", OpenAIChatCompletion(platform_model_name, openai_api_key, openai_org_id))
            print("You made a kernel using GPT model on OpenAI")
        
        # Setup Hugging Face kernel
        elif kernel_platform == "huggingface":
            kernel.add_text_completion_service("huggingface", HuggingFaceTextCompletion(platform_model_name, task="text-generation"))
            print("You made an open source kernel using an open source AI model on Hugging Face")

    # Return the kernel
    return(kernel)


# # Test the function
# # Read in OpenAI API key
# my_openai_api_key = open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# # Red in OpenAI org ID
# my_openai_org_id = open("C:/Users/oneno/OneDrive/Desktop/OpenAI Org ID.txt", "r").read()
# # Create kernel on OpenAI
# my_kernel = CreateSemanticKernel(
#     kernel_platform="openai",
#     openai_api_key=my_openai_api_key,
#     openai_org_id=my_openai_org_id
# )
