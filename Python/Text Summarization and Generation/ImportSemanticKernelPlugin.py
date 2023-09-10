import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion

# Declare function
def ImportSemanticKernelPlugin(kernel,
                         plugin_directory,
                         plugin_name):
    # Import the plugin
    semantic_kernel_plugin = kernel.import_semantic_skill_from_directory(plugin_directory, plugin_name)
    
    # Return the plugin
    return(semantic_kernel_plugin)


# # Test function. Run CreateSemanticKernel.py and CreateSemanticContext.py first
# plugin_IntelligenceAnalysis = ImportSemanticKernelPlugin(
#     kernel=my_kernel,
#     plugin_directory="C:/Users/oneno/OneDrive/Creations/Semantic Kernel Plugins/AnalysisToolBox-Plugins",
#     plugin_name="IntelligenceAnalysis"
# )
# # Run plugin as test
# swot_opportunities_result = await my_kernel.run_async(
#     plugin_IntelligenceAnalysis["SWOTAnalysisOpportunityStrategies"],
#     input_context=my_context
# )
# # Display result
# from IPython.display import display, Markdown
# display(Markdown(str(swot_opportunities_result)))
