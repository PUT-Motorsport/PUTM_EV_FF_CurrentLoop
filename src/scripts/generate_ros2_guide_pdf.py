#!/usr/bin/env python3
"""Convert docs/ros2_integration_guide.md to PDF via weasyprint."""
import markdown
from pathlib import Path
from weasyprint import HTML, CSS

DOCS_DIR = Path(__file__).parent.parent.parent / 'docs'
MD_PATH  = DOCS_DIR / "ros2_integration_guide.md"
PDF_PATH = DOCS_DIR / "ros2_integration_guide.pdf"

md_text = MD_PATH.read_text(encoding="utf-8")

body_html = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "nl2br"],
)

CSS_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

@page {
    size: A4;
    margin: 2cm 2.2cm 2.2cm 2.2cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #888;
        font-family: Inter, Arial, sans-serif;
    }
}

body {
    font-family: Inter, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1a1a1a;
}

h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #0d1b2a;
    margin-bottom: 4pt;
    padding-bottom: 8pt;
    border-bottom: 3px solid #1a73e8;
}

h2 {
    font-size: 13pt;
    font-weight: 700;
    color: #1a2940;
    margin-top: 20pt;
    margin-bottom: 6pt;
    padding-bottom: 3pt;
    border-bottom: 1px solid #cdd5e0;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 14pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

/* Meta block after h1 */
h1 + p {
    color: #555;
    font-size: 9.5pt;
    margin-bottom: 16pt;
}

p {
    margin: 5pt 0 8pt 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0 14pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th {
    background: #1a73e8;
    color: #ffffff;
    padding: 6pt 8pt;
    text-align: left;
    font-weight: 600;
}

td {
    padding: 5pt 8pt;
    border-bottom: 1px solid #e0e7ef;
    vertical-align: top;
}

tr:nth-child(even) td {
    background: #f4f7fc;
}

code {
    background: #f0f4f8;
    border: 1px solid #d0d9e6;
    border-radius: 3px;
    padding: 1pt 4pt;
    font-family: 'Courier New', monospace;
    font-size: 8.5pt;
}

pre {
    background: #f0f4f8;
    border: 1px solid #d0d9e6;
    border-left: 4px solid #1a73e8;
    border-radius: 4px;
    padding: 10pt 12pt;
    font-family: 'Courier New', monospace;
    font-size: 8pt;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-all;
    margin: 8pt 0 12pt 0;
    page-break-inside: avoid;
}

pre code {
    background: none;
    border: none;
    padding: 0;
    font-size: 8pt;
}

blockquote {
    border-left: 4px solid #f4a21a;
    background: #fffbf0;
    margin: 10pt 0;
    padding: 8pt 14pt;
    color: #4a3800;
    font-size: 9.5pt;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
}

ul, ol {
    margin: 6pt 0 8pt 0;
    padding-left: 18pt;
}

li {
    margin-bottom: 3pt;
}

strong {
    color: #0d1b2a;
}

hr {
    border: none;
    border-top: 1px solid #cdd5e0;
    margin: 18pt 0;
}
"""

full_html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Instrukcja wdrożenia predyktora prądów do ROS2</title>
</head>
<body>
{body_html}
</body>
</html>"""

HTML(string=full_html).write_pdf(
    PDF_PATH,
    stylesheets=[CSS(string=CSS_STYLES)],
)
print(f"PDF saved: {PDF_PATH}")
