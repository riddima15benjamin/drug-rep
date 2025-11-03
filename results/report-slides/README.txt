Report and Slides Directory
==============================

This directory contains the project report and presentation slides.

Files:
------

1. report.md
   - Full 2-page technical report in Markdown format
   - Covers: objective, data, model, results, interpretation, conclusions
   - Can be converted to PDF using: pandoc report.md -o report.pdf

2. demo-slides.md
   - 12-slide presentation deck in Markdown format
   - Suitable for project demonstrations and talks
   - Can be converted to PPTX using: pandoc demo-slides.md -o demo-slides.pptx

3. README.txt
   - This file

Conversion Instructions:
------------------------

To convert to PDF/PPTX, install Pandoc (https://pandoc.org/) and run:

# Generate PDF report
pandoc report.md -o report.pdf --pdf-engine=xelatex

# Generate PowerPoint slides
pandoc demo-slides.md -o demo-slides.pptx

Alternative tools:
- Markdown to PDF: grip, md-to-pdf
- Markdown to PPTX: Marp, reveal.js

Note: The slides include presenter notes and can be customized for your presentation style.

For best results with slides, use a tool that supports slide breaks (---).

