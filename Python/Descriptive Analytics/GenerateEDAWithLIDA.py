
# Import packages
from lida import llm, Manager, TextGenerationConfig
from lida.utils import plot_raster, plt
import openai
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Declare function
def GenerateEDAWithLIDA(dataframe,
                        llm_api_key,
                        llm_provider="openai",
                        llm_model="gpt-3.5-turbo",
                        visualization_library="seaborn",
                        goal_temperature=0.50,
                        code_generation_temperature=0.05,
                        number_of_goals_to_generate=5,
                        plot_recommended_visualization=False,
                        show_code_for_recommended_visualization=False):
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
        summary_method="default", 
        textgen_config=goal_textgen_config
    )
    goals = lida.goals(
        summary, 
        n=number_of_goals_to_generate,
        textgen_config=goal_textgen_config
    )

    # Create charts for goals
    for i in range(number_of_goals_to_generate):
        print("Exploratory Data Analysis Goal/Question #" + str(i+1) + ":", goals[i].question)
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


# # Test function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# iris['species'] = iris['species'].astype('category')
# GenerateEDAWithLIDA(
#     dataframe=iris,
#     llm_api_key = open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read(),
#     number_of_goals_to_generate=6
# )
