import semantic_kernel as sk
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion, OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
import shutil

# Declare function
def RunSemanticKernelPlugin(plugin_directory,
                            plugin_name,
                            plugin_module_name,
                            # Kernel parameters
                            kernel=None,
                            kernel_platform="openai",
                            kernel_platform_model_name="gpt-3.5-turbo",
                            openai_api_key=None,
                            openai_org_id=None,
                            # Context parameters
                            context=None,
                            # Memory store parameters
                            create_memory=False,
                            memory_store_name=None,
                            memory_text_completion_service="openai-completion",
                            memory_text_completion_model="gpt-3.5-turbo",
                            memory_text_embedding_service="openai-embedding",
                            memory_text_embedding_model="text-embedding-ada-002"):
    # If kernel is not specified, create a new one
    if kernel is None:
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
                kernel.add_text_completion_service("openai", OpenAIChatCompletion(kernel_platform_model_name, openai_api_key, openai_org_id))
                print("You made a kernel using GPT model on OpenAI")
            
            # Setup Hugging Face kernel
            elif kernel_platform == "huggingface":
                kernel.add_text_completion_service("huggingface", HuggingFaceTextCompletion(kernel_platform_model_name, task="text-generation"))
                print("You made an open source kernel using an open source AI model on Hugging Face")
    
    # If context is not specified, create a new one
    if context is not None:
        # Ensure that the context is a dictionary
        if not isinstance(context, dict):
            raise TypeError("Context must be a dictionary.")
        
        # Create a context object
        kernel_context = kernel.create_new_context()
        
        # Iterate through context dictionary and add to context object
        for key, value in context.items():
            kernel_context[key] = value

    # Import the plugin
    semantic_kernel_plugin = kernel.import_semantic_skill_from_directory(plugin_directory, plugin_name)
    
    # Create a memory store from the plugin use
    if create_memory:
        # If memory store name is not specified, use "memory_store"
        if memory_store_name is None:
            memory_store_name = "memory_store"
            
        # Set the chat completion service
        kernel.add_text_completion_service(
            memory_text_completion_service, 
            OpenAIChatCompletion(memory_text_completion_model, openai_api_key, openai_org_id)
        )
        
        # Set up the text embedding service 
        kernel.add_text_embedding_generation_service(
            memory_text_embedding_service, 
            OpenAITextEmbedding(memory_text_embedding_model, openai_api_key, openai_org_id)
        )
        
        # Create a memory store from the plugin use
        kernel.register_memory_store(memory_store=ChromaMemoryStore(persist_directory=memory_store_name))
        print("Made two new services attached to the kernel and made a Chroma memory store that's persistent.")
    
    # Run the plugin and get its output
    plugin_output = kernel.run_async(
        semantic_kernel_plugin[plugin_module_name],
        input_context=kernel_context
    )
    
    # Return the plugin's output
    return(plugin_output)


# # Test function. 
# # Read in OpenAI API key
# my_openai_api_key = open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# # Read in OpenAI org ID
# my_openai_org_id = open("C:/Users/oneno/OneDrive/Desktop/OpenAI Org ID.txt", "r").read()
# # Setup context
# strengths = [ "Unique garlic pizza recipe that wins top awards","Owner trained in Sicily at some of the best pizzerias","Strong local reputation","Prime location on university campus" ]
# weaknesses = [ "High staff turnover","Floods in the area damaged the seating areas that are in need of repair","Absence of popular calzones from menu","Negative reviews from younger demographic for lack of hip ingredients" ]
# opportunities = [ "Untapped catering potential","Growing local tech startup community","Unexplored online presence and order capabilities","Upcoming annual food fair" ]
# threats = [ "Competition from cheaper pizza businesses nearby","There's nearby street construction that will impact foot traffic","Rising cost of cheese will increase the cost of pizzas","No immediate local regulatory changes but it's election season" ]
# # Run SWOT plugin as test
# swot_opportunities_result = await RunSemanticKernelPlugin(
#     plugin_directory="C:/Users/oneno/OneDrive/Creations/Semantic Kernel Plugins/AnalysisToolBox-Plugins",
#     plugin_name="IntelligenceAnalysis",
#     plugin_module_name="SWOTAnalysisOpportunityStrategies",
#     context={
#         "input": "makes pizzas",
#         "strengths": ", ".join(strengths),
#         "weaknesses": ", ".join(weaknesses),
#         "opportunities": ", ".join(opportunities),
#         "threats": ", ".join(threats),
#     },
#     openai_api_key=my_openai_api_key,
#     openai_org_id=my_openai_org_id
# )
# from IPython.display import display, Markdown
# display(Markdown(str(swot_opportunities_result)))
