# Load packages
import os
import base64

# Declare function
def ConvertWordToHTML(docx_file_path, html_file_path=None):
    """
    Convert Word document to HTML with embedded images and preserved formatting.

    This function converts Microsoft Word documents (.docx) to clean, semantic HTML using the
    Mammoth library. It preserves document structure including headings, paragraphs, tables,
    lists, and formatting while embedding images as base64-encoded data URIs for self-contained
    HTML output. The function automatically converts footnotes to endnotes, adds responsive CSS
    styling, and handles path resolution intelligently. The resulting HTML is web-ready and
    requires no external image files.

    Word-to-HTML conversion is essential for:
      * Web publishing and content management system migration
      * Creating accessible, responsive documentation from Word files
      * Email newsletter generation with embedded content
      * Archiving documents in open, platform-independent format
      * Integrating Word content into web applications and portals
      * Creating self-contained HTML reports and documentation
      * Converting legacy Word documents to modern web formats
      * Enabling full-text search and indexing of document content

    The function uses Mammoth for high-quality semantic conversion, embedding images as base64
    to eliminate external dependencies. It includes comprehensive error handling for file
    permissions, path validation, and OneDrive sync conflicts. The output HTML includes
    responsive CSS with proper table formatting, image scaling, and endnote styling. Automatic
    path handling allows flexible output specification—provide a full path, directory, or let
    the function use the source directory.

    Parameters
    ----------
    docx_file_path
        Absolute or relative path to the input Word document (.docx format). The file must
        exist and be readable. Older .doc format is not supported.
    html_file_path
        Path where the output HTML file will be saved. Can be:
          * Full file path ending in .html (e.g., 'output/report.html')
          * Directory path (function adds filename from source)
          * None (saves to same folder as input with .html extension)
        The function automatically creates missing directories and adds .html extension if
        needed. Defaults to None.

    Returns
    -------
    None
        This function does not return a value. It creates an HTML file at the specified
        location and prints status messages including warnings from the conversion process.

    Raises
    ------
    FileNotFoundError
        If the input .docx file does not exist at the specified path.
    ValueError
        If the input file is not a .docx file or if the output path is invalid.
    PermissionError
        If the function lacks write permissions to the output location, if the output file
        is open in another application, or if OneDrive sync prevents file access.

    Notes
    -----
    * **Image Embedding**: All images are embedded as base64 data URIs, increasing file size
      but eliminating external dependencies
    * **Footnote Conversion**: All footnotes are automatically converted to endnotes in the
      HTML output
    * **CSS Styling**: Includes responsive CSS for tables, images, and general formatting
    * **Path Flexibility**: Intelligently handles various path formats and creates directories
    * **Encoding**: Output uses UTF-8 encoding for international character support

    Examples
    --------
    # Basic conversion with automatic output path
    ConvertWordToHTML('report.docx')
    # Creates 'report.html' in the same folder as the input file

    # Specify custom output location
    ConvertWordToHTML(
        docx_file_path='C:/Documents/Annual_Report.docx',
        html_file_path='C:/Website/reports/annual_report.html'
    )
    # Converts to specified path, creates directories if needed

    # Output to directory (filename derived from input)
    ConvertWordToHTML(
        docx_file_path='proposals/client_proposal.docx',
        html_file_path='web_content/'
    )
    # Creates 'web_content/client_proposal.html'

    # Batch conversion workflow
    import os
    import glob
    
    # Convert all Word docs in a folder
    source_dir = 'word_documents'
    output_dir = 'html_output'
    
    for docx_file in glob.glob(f'{source_dir}/*.docx'):
        ConvertWordToHTML(docx_file, output_dir)
    # Converts all .docx files to HTML in output directory

    # Web publishing with automatic path handling
    ConvertWordToHTML(
        docx_file_path='content/blog_post_2024.docx',
        html_file_path='website/blog/'
    )
    # Creates 'website/blog/blog_post_2024.html' with embedded images

    """
    # Lazy load uncommon packages
    import mammoth
    
    # Check if input file exists
    if not os.path.exists(docx_file_path):
        raise FileNotFoundError(f"Input file not found: {docx_file_path}")
    
    # Check if input file is .docx
    if not docx_file_path.lower().endswith('.docx'):
        raise ValueError("Input file must be a .docx file")
    
    # If html_file_path is None, use the same folder as the input file
    if html_file_path is None:
        input_dir = os.path.dirname(docx_file_path)
        input_filename = os.path.splitext(os.path.basename(docx_file_path))[0]
        html_file_path = os.path.join(input_dir, f"{input_filename}.html")
        print(f"No output path specified. Using: {html_file_path}")
    
    # Check if html_file_path is a directory instead of a file
    if os.path.isdir(html_file_path):
        # If it's a directory, create a filename based on the input file
        input_filename = os.path.splitext(os.path.basename(docx_file_path))[0]
        html_file_path = os.path.join(html_file_path, f"{input_filename}.html")
        print(f"Directory provided instead of file path. Using: {html_file_path}")
    
    # Check if html_file_path ends with a path separator (indicating it's meant to be a directory)
    if html_file_path.endswith(('/', '\\')):
        input_filename = os.path.splitext(os.path.basename(docx_file_path))[0]
        html_file_path = os.path.join(html_file_path, f"{input_filename}.html")
        print(f"Directory path provided. Using: {html_file_path}")
    
    # Ensure the output file has .html extension
    if not html_file_path.lower().endswith('.html'):
        html_file_path += '.html'
        print(f"Added .html extension. Using: {html_file_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(html_file_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except PermissionError:
            raise PermissionError(f"Permission denied: Cannot create directory '{output_dir}'. "
                                f"Please check your permissions or choose a different output location.")
    
    # Test write access to the output location
    try:
        test_file = html_file_path + '.tmp'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except PermissionError:
        raise PermissionError(f"Permission denied: Cannot write to location '{html_file_path}'. "
                            f"Please check that:\n"
                            f"1. The file is not currently open in another application\n"
                            f"2. You have write permissions to this directory\n"
                            f"3. The path is not read-only\n"
                            f"4. OneDrive sync is not preventing file access")
    except OSError as e:
        raise ValueError(f"Invalid output path '{html_file_path}': {str(e)}")
    
    # Define custom image converter to embed images as base64
    def convert_image(image):
        with image.open() as image_bytes:
            encoded_src = base64.b64encode(image_bytes.read()).decode("ascii")
        return {
            "src": f"data:{image.content_type};base64,{encoded_src}"
        }
    
    # Define style mapping to ensure footnotes are converted to endnotes
    # and preserve other formatting
    style_map = """
    p[style-name='Footnote Text'] => p.endnote
    p[style-name='Endnote Text'] => p.endnote
    comment-reference => sup
    """
    
    try:
        # Open and convert the docx file
        with open(docx_file_path, "rb") as docx_file:
            result = mammoth.convert_to_html(
                docx_file,
                convert_image=mammoth.images.img_element(convert_image),
                style_map=style_map
            )
        
        # Get the HTML content
        html_content = result.value
        
        # Add basic HTML structure and CSS for better formatting
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Converted Document</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .endnote {{
            font-size: 0.9em;
            margin-top: 2em;
            border-top: 1px solid #ccc;
            padding-top: 1em;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        
        # Write the HTML content to the output file
        try:
            with open(html_file_path, 'w', encoding='utf-8') as html_file:
                html_file.write(full_html)
        except PermissionError:
            raise PermissionError(f"Permission denied: Cannot write to '{html_file_path}'. "
                                f"The file may be open in another application, or you may not have "
                                f"write permissions to this location. Please close any open files "
                                f"and check your permissions.")
        except OSError as e:
            if "Invalid argument" in str(e) or "cannot find the path" in str(e).lower():
                raise ValueError(f"Invalid file path: '{html_file_path}'. "
                               f"Please ensure the path is valid and accessible.")
            else:
                raise OSError(f"Error writing file: {str(e)}")
        
        # Print any warnings from mammoth
        if result.messages:
            print("Conversion warnings:")
            for message in result.messages:
                print(f"  - {message}")
        
        print(f"Successfully converted {docx_file_path} to {html_file_path}")
        
    except Exception as e:
        raise Exception(f"Error converting document: {str(e)}")