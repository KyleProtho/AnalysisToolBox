# Load packages
import os

# Declare function
def CreateFunctionForChatGPT(function_name,
                             function_description,
                             json_output,
                             list_of_required_arguments=[],
                             parameter_type="object",
                             function_file_path=None):
    # Create the function JSON object
    custom_function = {
        "name": function_name,
        "description": function_description,
        "parameters": {
            "type": parameter_type,
            "properties": json_output
        }
    }
    
    # If list_of_required_arguments is not empty, then add it to the function JSON object in the "parameters" key
    if len(list_of_required_arguments) > 0:
        # Ensure that that the items in the list match the keys in the json_output
        for required_argument in list_of_required_arguments:
            if required_argument not in json_output.keys():
                raise ValueError('The list of required arguments must match the keys in the json_output.')
        # Add the list_of_required_arguments to the function JSON object
        custom_function["parameters"]["required"] = list_of_required_arguments
    print("Function called " + function_name + " created successfully.")
    
    # If function_file_path is not None, then write the function to the file
    if function_file_path is not None:
        # Ensure the file ends with .json
        if function_file_path[-5:] != '.json':
            raise ValueError('The function file path must end with .json')
        else:
            with open(function_file_path, 'w') as f:
                # Replace single quotes with double quotes
                custom_function = str(custom_function).replace("'", '"')
                # Write the function to the file
                f.write(str(custom_function))
                print("Function called " + function_name + " written to " + function_file_path + " successfully.")
    
    # Return the function JSON object
    return custom_function


# # Test the function
# student_info_extraction = CreateFunctionForChatGPT(
#     function_name = "extract_student_info",
#     function_description = "Get the student information from the body of the input text",
#     json_output = {
#         "name": {
#             'type': 'string',
#             'description': 'Name of the person'
#         },
#         'major': {
#             'type': 'string',
#             'description': 'Major subject.'
#         },
#         'school': {
#             'type': 'string',
#             'description': 'The university name.'
#         },
#         'grades': {
#             'type': 'integer',
#             'description': 'GPA of the student.'
#         },
#         'club': {
#             'type': 'string',
#             'description': 'School club for extracurricular activities. '
#         }
#     },
#     list_of_required_arguments=["name"],
#     function_file_path="C:/Users/oneno/Downloads/student_info_extraction.json"
# )
