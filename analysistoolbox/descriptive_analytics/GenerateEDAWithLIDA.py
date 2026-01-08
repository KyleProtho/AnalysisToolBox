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
    Generate AI-powered exploratory data analysis goals using Microsoft LIDA.

    This function leverages Microsoft's LIDA (Automatic Generation of Visualizations and
    Infographics using Large Language Models) to automatically analyze datasets and generate
    meaningful exploratory data analysis goals. LIDA uses large language models to understand
    data structure, identify interesting patterns, and recommend relevant visualizations with
    executable code. The function summarizes the dataset, generates analytical questions, and
    optionally produces visualization code and plots, accelerating the EDA process.

    AI-powered EDA with LIDA is essential for:
      * Rapid initial data exploration and hypothesis generation
      * Automated insight discovery in unfamiliar datasets
      * Generating visualization ideas for complex multivariate data
      * Reducing time-to-insight for data analysts and scientists
      * Creating reproducible EDA workflows with generated code
      * Educational purposes to learn effective EDA techniques
      * Augmenting human analysis with AI-suggested perspectives
      * Standardizing EDA practices across teams and projects

    The function uses LLMs to create a semantic understanding of the data, generating natural
    language descriptions of columns, identifying data types and distributions, and proposing
    analytical questions with rationale. It can generate visualization code in multiple libraries
    (seaborn, matplotlib, plotly) and optionally execute and display the visualizations. Temperature
    parameters control the creativity of goal generation versus the precision of code generation.

    Parameters
    ----------
    dataframe
        A pandas DataFrame to analyze. LIDA will examine its structure, data types, distributions,
        and relationships to generate relevant EDA goals.
    llm_api_key
        API key for the LLM provider (e.g., OpenAI API key). Required for authentication with
        the language model service.
    llm_provider
        Name of the LLM provider to use. Supported options include 'openai', 'azure-openai',
        'palm', 'cohere', and others supported by LIDA. Defaults to 'openai'.
    llm_model
        Specific model name to use from the provider. For OpenAI: 'gpt-3.5-turbo', 'gpt-4',
        'gpt-4-turbo'. More capable models generate better insights but cost more. Defaults to
        'gpt-3.5-turbo'.
    visualization_library
        Python visualization library for generated code. Options: 'seaborn', 'matplotlib',
        'plotly', 'altair', 'ggplot'. Defaults to 'seaborn'.
    goal_temperature
        Temperature parameter (0-1) for goal generation. Higher values (0.7-1.0) produce more
        creative/diverse goals; lower values (0.1-0.3) produce more focused/conservative goals.
        Defaults to 0.50.
    code_generation_temperature
        Temperature parameter (0-1) for code generation. Lower values (0.0-0.2) produce more
        deterministic, reliable code; higher values may introduce variability. Defaults to 0.05.
    data_summary_method
        Method for data summarization. 'llm' uses the language model for semantic understanding;
        'default' uses statistical methods. LLM method provides richer insights. Defaults to 'llm'.
    number_of_samples_to_show_in_summary
        Number of sample rows to include in the data summary sent to the LLM. More samples
        provide better context but increase token usage. Defaults to 5.
    return_data_fields_summary
        Whether to return a DataFrame containing the LLM-generated field descriptions and
        properties. Useful for understanding how LIDA interpreted the data. Defaults to True.
    number_of_goals_to_generate
        Number of EDA goals/questions to generate. Each goal includes a question, rationale,
        and recommended visualization. More goals provide broader coverage. Defaults to 5.
    plot_recommended_visualization
        Whether to execute the generated visualization code and display the plots. Requires
        the specified visualization library to be installed. Defaults to False.
    show_code_for_recommended_visualization
        Whether to print the generated Python code for each visualization. Useful for learning
        and customization. Defaults to False.

    Returns
    -------
    pd.DataFrame or None
        If return_data_fields_summary is True, returns a DataFrame with columns:
          * description: LLM-generated semantic description of each field
          * dtype: Detected data type
          * samples: Example values from the field
          * Additional properties like min, max, mean for numeric fields
        If False, returns None. The function prints EDA goals and visualizations to stdout.

    Examples
    --------
    # Basic EDA goal generation with OpenAI
    import pandas as pd
    sales_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'product': ['A', 'B', 'C'] * 33 + ['A'],
        'revenue': [100 + i * 10 for i in range(100)],
        'units_sold': [10 + i % 20 for i in range(100)]
    })
    field_summary = GenerateEDAWithLIDA(
        sales_df,
        llm_api_key='sk-...',
        llm_model='gpt-4',
        number_of_goals_to_generate=3
    )
    # Generates 3 analytical questions with rationale

    # Generate goals with visualization code
    customer_df = pd.DataFrame({
        'customer_id': range(1, 201),
        'age': [25 + i % 50 for i in range(200)],
        'spend': [100 + i * 5 for i in range(200)],
        'segment': ['Premium', 'Standard', 'Basic'] * 66 + ['Premium', 'Standard']
    })
    summary = GenerateEDAWithLIDA(
        customer_df,
        llm_api_key='sk-...',
        llm_provider='openai',
        llm_model='gpt-3.5-turbo',
        visualization_library='seaborn',
        number_of_goals_to_generate=5,
        show_code_for_recommended_visualization=True,
        goal_temperature=0.7
    )
    # Shows generated seaborn code for each visualization

    # Full EDA with plotted visualizations
    marketing_df = pd.DataFrame({
        'campaign': ['Email', 'Social', 'Search'] * 50,
        'impressions': [1000 + i * 100 for i in range(150)],
        'clicks': [50 + i * 2 for i in range(150)],
        'conversions': [5 + i % 10 for i in range(150)],
        'cost': [100 + i * 5 for i in range(150)]
    })
    fields = GenerateEDAWithLIDA(
        marketing_df,
        llm_api_key='sk-...',
        llm_model='gpt-4-turbo',
        visualization_library='plotly',
        number_of_goals_to_generate=4,
        plot_recommended_visualization=True,
        show_code_for_recommended_visualization=True,
        goal_temperature=0.6,
        code_generation_temperature=0.1,
        number_of_samples_to_show_in_summary=10
    )
    # Generates and displays 4 plotly visualizations with code

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

