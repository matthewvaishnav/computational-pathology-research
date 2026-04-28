#!/usr/bin/env python3
"""
Convert Harvard-style resume from Markdown to PDF using Python libraries.
"""

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    import re
    
    def markdown_to_pdf():
        """Convert markdown resume to PDF using ReportLab."""
        
        # Read the markdown file
        with open("Matthew_Vaishnav_Harvard_Style_Resume.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Create PDF
        doc = SimpleDocTemplate(
            "Matthew_Vaishnav_Harvard_Style_Resume.pdf",
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=6,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=4,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            fontName='Helvetica'
        )
        
        # Parse content
        story = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
                continue
                
            # Title (# MATTHEW VAISHNAV)
            if line.startswith('# '):
                text = line[2:].strip()
                story.append(Paragraph(text, title_style))
                
            # Main headings (## SECTION)
            elif line.startswith('## '):
                text = line[3:].strip()
                story.append(Paragraph(text, heading_style))
                
            # Subheadings (### Project Name)
            elif line.startswith('### '):
                text = line[4:].strip()
                # Remove markdown formatting
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                story.append(Paragraph(text, subheading_style))
                
            # Horizontal rules
            elif line.startswith('---'):
                story.append(Spacer(1, 12))
                
            # Bullet points
            elif line.startswith('• '):
                text = line[2:].strip()
                # Convert markdown formatting to HTML
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
                story.append(Paragraph(f"• {text}", body_style))
                
            # Regular text
            elif line and not line.startswith('*') and not line.startswith('_'):
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
                story.append(Paragraph(text, body_style))
        
        # Build PDF
        doc.build(story)
        print("✅ Successfully created Matthew_Vaishnav_Harvard_Style_Resume.pdf")
        return True

    if __name__ == "__main__":
        markdown_to_pdf()

except ImportError:
    print("❌ ReportLab not installed!")
    print("Install with: pip install reportlab")
    print("\nAlternatively, you can:")
    print("1. Install pandoc: https://pandoc.org/installing.html")
    print("2. Use online converters like markdown-pdf.com")
    print("3. Copy the markdown content to Google Docs and export as PDF")