import json
import re
import os

def export_notebook_to_html(notebook_path, output_path):
    """Export Jupyter notebook to HTML without code cells"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Basic HTML template
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Swimming of P. dumerilii larva</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .cell { margin-bottom: 20px; }
        .markdown-cell { }
        .output-cell { margin: 10px 0; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #34495e; }
        img { max-width: 100%; height: auto; }
        pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .math { font-family: "Times New Roman", serif; }
        .output-text { background-color: #f8f9fa; padding: 10px; border-left: 4px solid #3498db; }
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
    </script>
</head>
<body>
"""
    
    html_content = html_template
    
    # Process each cell
    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'markdown':
            # Process markdown cell
            source = ''.join(cell.get('source', []))
            html_content += f'<div class="cell markdown-cell">{markdown_to_html(source)}</div>\n'
        
        elif cell['cell_type'] == 'code':
            # Skip code input, but include outputs
            outputs = cell.get('outputs', [])
            if outputs:
                html_content += '<div class="cell output-cell">\n'
                for output in outputs:
                    html_content += process_output(output)
                html_content += '</div>\n'
    
    html_content += """
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Successfully exported to {output_path}")

def markdown_to_html(markdown_text):
    """Convert markdown to HTML (basic implementation)"""
    # Convert headers
    markdown_text = re.sub(r'^# (.*)', r'<h1>\1</h1>', markdown_text, flags=re.MULTILINE)
    markdown_text = re.sub(r'^## (.*)', r'<h2>\1</h2>', markdown_text, flags=re.MULTILINE)
    markdown_text = re.sub(r'^### (.*)', r'<h3>\1</h3>', markdown_text, flags=re.MULTILINE)
    
    # Convert italics
    markdown_text = re.sub(r'_([^_]+)_', r'<em>\1</em>', markdown_text)
    
    # Convert bold
    markdown_text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', markdown_text)
    
    # Convert math expressions (keep as is for MathJax)
    # markdown_text = re.sub(r'\$\$(.*?)\$\$', r'<div class="math">$$\1$$</div>', markdown_text, flags=re.DOTALL)
    # markdown_text = re.sub(r'\$([^$]+)\$', r'<span class="math">$\1$</span>', markdown_text)
    
    # Convert line breaks
    markdown_text = markdown_text.replace('\n\n', '</p><p>')
    markdown_text = markdown_text.replace('\n', '<br>')
    
    # Wrap in paragraphs
    if markdown_text.strip():
        markdown_text = f'<p>{markdown_text}</p>'
    
    return markdown_text

def process_output(output):
    """Process notebook output"""
    output_html = ""
    
    if output.get('output_type') == 'display_data' or output.get('output_type') == 'execute_result':
        data = output.get('data', {})
        
        # Handle images
        if 'image/png' in data:
            img_data = data['image/png']
            if isinstance(img_data, list):
                img_data = ''.join(img_data)
            output_html += f'<img src="data:image/png;base64,{img_data}" alt="Plot">\n'
        
        # Handle text output
        if 'text/plain' in data:
            text_data = data['text/plain']
            if isinstance(text_data, list):
                text_data = ''.join(text_data)
            output_html += f'<div class="output-text"><pre>{text_data}</pre></div>\n'
    
    elif output.get('output_type') == 'stream':
        text = output.get('text', '')
        if isinstance(text, list):
            text = ''.join(text)
        output_html += f'<div class="output-text"><pre>{text}</pre></div>\n'
    
    return output_html

# Change to the directory containing the notebook
os.chdir('/home/pz723/cuda-filaments/analysis')

# Export the notebook
try:
    export_notebook_to_html('wavelength_swimming.ipynb', 'wavelength_swimming.html')
    print("HTML export completed successfully!")
except Exception as e:
    print(f"Error: {e}")