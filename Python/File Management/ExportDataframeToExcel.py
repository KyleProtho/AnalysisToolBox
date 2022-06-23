# Load packages
import pandas as pd
import xlsxwriter

# Declare function
def ExportDataframeToExcel(dictionary_sheetname_to_dataframe,
                           file_path,
                           include_index = True):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    for key, value in dictionary_sheetname_to_dataframe.items():
        current_df = dictionary_sheetname_to_dataframe[key]
        current_df.to_excel(writer, 
                            sheet_name = key,
                            index = include_index)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
