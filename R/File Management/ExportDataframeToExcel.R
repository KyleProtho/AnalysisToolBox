# Load packages
library(xlsx)

# Declare function
ExportDataframeToExcel = function(list_of_dataframes,
                                  file_name) {
  # Ensure that list of dataframes is list type
  if (typeof(list_of_dataframes) != "list") {
    stop("Put your dataframe in a list using the list() function.")
  }
  
  # Create workbook object
  wb = createWorkbook(type="xlsx")
  
  # Set data cell style
  cell_style_data = CellStyle(wb) +
    Font(
      wb,
      color = "#383838",
      name = "Arial",
      heightInPoints = 10
    )
  
  # Set column header style
  cell_style_column_header = CellStyle(wb) + 
    Border(
      color = "#d9dbda", 
      position = "BOTTOM", 
      pen = "BORDER_THICK"
    ) + 
    Font(
      wb,
      heightInPoints = 12,
      isBold = TRUE
    ) +
    Alignment(
      wrapText = TRUE,
      vertical = "VERTICAL_BOTTOM"
    )
  
  # Iterate through list of dataframes and write on separate sheets
  for (i in 1:length(list_of_dataframes)) {
    df_current = list_of_dataframes[i]
    df_current = df_current[[1]]
    
    # Create sheet in wb
    name_of_sheet = paste0("Sheet", i)
    sheet = createSheet(wb, sheetName = name_of_sheet)
    
    # Write dataframe
    addDataFrame(df_current, 
                 sheet, 
                 startRow = 1, 
                 startColumn = 1,
                 colnamesStyle = cell_style_column_header,
                 row.names = FALSE)
    
    # Set column width
    setColumnWidth(sheet, 
                   colIndex = 1:ncol(df_current),
                   colWidth = 15)
    
    # Freeze column name pane
    createFreezePane(sheet = sheet, 
                     rowSplit = 2,
                     colSplit = 1,
                     startRow = 2, 
                     startColumn = 1)
  }
  
  # Save workbook at specified file location
  saveWorkbook(wb, file_name)
}
