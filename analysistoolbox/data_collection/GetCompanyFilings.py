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
    Search and retrieve company filings from the SEC EDGAR database.

    This function provides programmatic access to the U.S. Securities and Exchange Commission's
    EDGAR (Electronic Data Gathering, Analysis, and Retrieval) system, which contains financial
    and business disclosure documents filed by public companies. The function wraps the EDGAR
    tool to search for specific filings based on keywords, date ranges, filing types, and
    company identifiers.

    EDGAR filings are essential for:
      * Financial analysis and investment research
      * Due diligence and corporate intelligence
      * Regulatory compliance monitoring
      * Competitive analysis and market research
      * Risk assessment and fraud detection
      * Academic research on corporate behavior
      * Tracking mergers, acquisitions, and corporate actions

    The function supports filtering by multiple criteria including company CIK (Central Index Key),
    filing type (e.g., 10-K, 10-Q, 8-K), date range, and company location. Results are returned
    as a pandas DataFrame for easy analysis and can optionally be saved to CSV for archival.

    Parameters
    ----------
    search_keywords
        Keywords to search for within filing documents. Use empty string for no keyword filter.
        Supports boolean operators and phrase searching.
    start_date
        Start date for the search period in 'YYYY-MM-DD' format. Only filings submitted on or
        after this date will be included.
    end_date
        End date for the search period in 'YYYY-MM-DD' format. Only filings submitted on or
        before this date will be included.
    filing_type
        Type of SEC filing to search for (e.g., '10-K', '10-Q', '8-K', 'DEF 14A'). Use 'all'
        to search across all filing types. Defaults to 'all'.
    company_cik
        Central Index Key (CIK) number to filter results to a specific company. Use empty
        string for no company filter. Defaults to '' (no filter).
    filter_by_location
        Type of location filter to apply. Must be either 'Incorporated in' or
        'Principal executive offices in'. Use empty string for no location filter.
        Defaults to '' (no filter).
    location
        Specific location (state or country) to filter by when using filter_by_location.
        Use empty string for no location filter. Defaults to '' (no filter).
    results_csv_filepath
        File path where search results should be saved as CSV. If None, a timestamped
        filename will be automatically generated. Defaults to None.
    results_log_filepath
        File path where search logs should be saved. If None, a timestamped log file
        will be automatically generated. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing search results with columns for company name, CIK, filing type,
        filing date, and document links. Returns empty DataFrame if no results are found.

    Examples
    --------
    # Search for all 10-K filings mentioning 'artificial intelligence' in 2023
    filings = GetCompanyFilings(
        search_keywords='artificial intelligence',
        start_date='2023-01-01',
        end_date='2023-12-31',
        filing_type='10-K'
    )

    # Search for all filings from a specific company using CIK
    apple_filings = GetCompanyFilings(
        search_keywords='',
        start_date='2022-01-01',
        end_date='2023-12-31',
        company_cik='0000320193',  # Apple Inc.
        filing_type='all'
    )

    # Search for 8-K filings from companies incorporated in Delaware
    delaware_8k = GetCompanyFilings(
        search_keywords='merger',
        start_date='2023-01-01',
        end_date='2023-12-31',
        filing_type='8-K',
        filter_by_location='Incorporated in',
        location='DE'
    )

    # Save results to a custom location
    filings = GetCompanyFilings(
        search_keywords='cybersecurity',
        start_date='2023-01-01',
        end_date='2023-12-31',
        filing_type='10-Q',
        results_csv_filepath='cybersecurity_filings.csv',
        results_log_filepath='search_log.log'
    )

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
