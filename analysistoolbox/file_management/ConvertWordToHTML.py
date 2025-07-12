# Load packages
import os
import base64

# Declare function
def ConvertWordToHTML(docx_file_path, html_file_path=None):
    """
    This function converts a Word document (.docx) to HTML format while preserving formatting,
    including headings, tables, images, footnotes, and endnotes. All footnotes are converted to endnotes.

    Args:
        docx_file_path (str): The path to the input .docx file.
        html_file_path (str, optional): The path where the output .html file will be saved.
                                       If None, saves to the same folder as the input file.
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