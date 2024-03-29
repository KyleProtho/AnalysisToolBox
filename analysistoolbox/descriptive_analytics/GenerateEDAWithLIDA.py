# Import packages
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Declare function
def GenerateEDAWithLIDA(dataframe,
                        llm_api_key,
                        # LLM parameters
                        llm_provider="openai",
                        llm_model="gpt-3.5-turbo",
                        visualization_library="seaborn",
                        goal_temperature=0.50,
                        code_generation_temperature=0.05,
                        # Data summarization parameters
                        data_summary_method="llm",
                        number_of_samples_to_show_in_summary=5,
                        return_data_fields_summary=True,
                        # EDA parameters
                        number_of_goals_to_generate=5,
                        plot_recommended_visualization=False,
                        show_code_for_recommended_visualization=False):
    """
    This function generates exploratory data analysis (EDA) goals using the LIDA package from Microsoft.
    
    Args:
        dataframe (Pandas dataframe): Pandas dataframe containing the data to be analyzed.
        llm_api_key (str): The API key for the LLM provider.
        llm_provider (str): The LLM provider. Defaults to "openai".
        llm_model (str): The LLM model. Defaults to "gpt-3.5-turbo".
        visualization_library (str): The visualization library to use. Defaults to "seaborn".
        goal_temperature (float, optional): The temperature to use for goal generation. Defaults to 0.50.
        code_generation_temperature (float, optional): The temperature to use for code generation. Defaults to 0.05.
        data_summary_method (str, optional): The method to use for data summarization. Defaults to "llm".
        number_of_samples_to_show_in_summary (int, optional): The number of samples to show in the data summary. Defaults to 5.
        return_data_fields_summary (bool, optional): Whether to return the data fields summary. Defaults to True.
        number_of_goals_to_generate (int, optional): The number of goals to generate. Defaults to 5.
        plot_recommended_visualization (bool, optional): Whether to plot the recommended visualization. Defaults to False.
        show_code_for_recommended_visualization (bool, optional): Whether to show the code for the recommended visualization. Defaults to False.

    Returns:
        Pandas dataframe: A dataframe containing the data fields summary.
    """
    # Lazy load uncommon packages
    from lida import llm, Manager, TextGenerationConfig
    from lida.utils import plot_raster, plt
    import openai
    
    # Set up LIDA and text generation model
    lida = Manager(text_gen=llm(
        provider=llm_provider,
        api_key=llm_api_key,
    )) 

    # Set up text generation config for goal generation
    goal_textgen_config = TextGenerationConfig(
        n=1, 
        temperature=goal_temperature, 
        model=llm_model, 
        use_cache=True
    )

    # Set up text generation config for code generation
    code_textgen_config = TextGenerationConfig(
        n=1,
        temperature=code_generation_temperature,
        model=llm_model,
        use_cache=True
    )

    # Create summary and goals
    summary = lida.summarize(
        data=dataframe, 
        summary_method=data_summary_method,
        textgen_config=goal_textgen_config,
        n_samples=number_of_samples_to_show_in_summary
    )
    goals = lida.goals(
        summary, 
        n=number_of_goals_to_generate,
        textgen_config=goal_textgen_config
    )
    
    # Print summary of the data
    if data_summary_method == "llm":
        print("-----  DATA SUMMARY  -----")
        # Get name of dataset from summary JSON
        print("Name:", summary['name'])
        print("Description:", summary['dataset_description'])
        
        # Convert fields to dataframe
        fields = pd.DataFrame(summary['fields'])
        
        # The properties column is a dictionary, so convert it to columns
        properties = pd.DataFrame(fields['properties'].tolist())
        fields = pd.concat([fields, properties], axis=1)
        fields = fields.drop(columns=['properties'])
        
        # Set column as index
        fields = fields.set_index('column')
        
        # Move description column to the front
        fields = fields[['description'] + [col for col in fields.columns if col != 'description']]
            

    # Create charts for goals
    print("-----  EXPLORATORY DATA ANALYSIS (EDA) GOALS  -----")
    for i in range(number_of_goals_to_generate):
        print("Question #" + str(i+1) + ":", goals[i].question)
        print("Why:", goals[i].rationale)
        if plot_recommended_visualization or show_code_for_recommended_visualization:
            # Generate charts
            charts = lida.visualize(
                summary=summary, 
                goal=goals[i], 
                textgen_config=goal_textgen_config, 
                library=visualization_library
            )
            
            # Show the plot if requested
            if plot_recommended_visualization:
                plot_raster(charts[0].raster)
                
            # Show the code if requested
            if show_code_for_recommended_visualization:
                print("Starter code for recommended visualization:", "\n\n", charts[0].code)
        else:
            # Describe the recommended visualization
            print("Recommended visulization:", goals[i].visualization)
        print("\n")
    
    # Return data summary dataframe if requested
    if return_data_fields_summary:
        return fields

