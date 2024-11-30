# Load common packages
import pandas as pd
from datetime import datetime

# Declare function to search EDGAR filings from the SEC
def GetCompanyFilings(search_keywords: str, 
                      start_date: str, 
                      end_date: str, 
                      filing_type: str = "all", 
                      company_cik: str = "",
                      filter_by_location: str = "", 
                      location: str = "",
                      results_csv_filepath = None,
                      results_log_filepath = None) -> pd.DataFrame:
    """
    Search EDGAR filings based on provided parameters. This function uses the EDGAR tool from the SEC to search for filings.

    Parameters:
        search_keywords (str): Keywords to search for in filings.
        start_date (str): Start date for the search in 'YYYY-MM-DD' format.
        end_date (str): End date for the search in 'YYYY-MM-DD' format.
        filing_type (str): Type of filings to search. Default is "all".
        company_cik (str): Company CIK to filter results. Default is "" (no filter).
        filter_by_location (str): Type of location filter, either "Incorporated in" or 
                                  "Principal executive offices in". Default is "" (no filter).
        location (str): Specific location to filter by. Default is "" (no filter).

    Returns:
        pd.DataFrame: A DataFrame containing the search results.
    """
    # Lazy load the EDGAR tool to avoid unnecessary imports
    from edgar_tool.cli import SecEdgarScraperCli as edgar_tool
    from edgar_tool.page_fetcher import NoResultsFoundError
    
    # Ensure that filter_by_location is valid
    if filter_by_location not in ["Incorporated in", "Principal executive offices in", ""]:
        raise ValueError("filter_by_location must be either 'Incorporated in' or 'Principal executive offices in'.")
    
    # Ensure that results_output_csv is None or ends with .csv
    if results_csv_filepath is not None and not results_csv_filepath.endswith(".csv"):
        raise ValueError("results_csv_filepath must be None or end with '.csv'.")
    
    # Ensure that results_log is None or ends with .log
    if results_log_filepath is not None and not results_log_filepath.endswith(".log"):
        raise ValueError("results_log_filepath must be None or end with '.log'.")
    
    # Handle empty parameters
    search_keywords = "\"\"" if search_keywords == "" else search_keywords
    filing_type = None if filing_type == "all" else filing_type
    company_cik = None if company_cik == "" else company_cik
    
    # Prepare location filter
    loc_filter = None
    if filter_by_location == "Incorporated in":
        loc_filter = {"inc_in": location}
    elif filter_by_location == "Principal executive offices in":
        loc_filter = {"peo_in": location}

    # Prepare output file paths
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    if results_csv_filepath is not None:
        output = results_csv_filepath
    else:
        output = f"edgar_search_results_{timestamp}.csv"
    if results_log_filepath is not None:
        logfile = results_log_filepath
    else:
        logfile = f"./edgar_log_{timestamp}.log"

    try:
        # Redirect logs to a file
        with open(logfile, 'a') as log_file:
            # Perform the EDGAR search
            edgar_tool.text_search(
                search_keywords,
                start_date=start_date,
                end_date=end_date,
                filing_form=filing_type,
                entity_id=company_cik,
                output=output,
                **(loc_filter or {})
            )
        print("Search completed successfully.")
        
        # Load results into a DataFrame
        results = pd.read_csv(output)
        return results

    except NoResultsFoundError:
        print("No results were found for your query.")
        return pd.DataFrame()
    except FileNotFoundError as e:
        print("Something went wrong with the EDGAR tool. Please check the logs for more details.")
        raise
    except Exception as e:
        print("An error occurred during the EDGAR search.")
        raise
