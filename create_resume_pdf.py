#!/usr/bin/env python3
"""
Convert Harvard-style resume from Markdown to PDF.
"""

import subprocess
import sys
from pathlib import Path

def create_pdf():
    """Convert markdown resume to PDF using pandoc."""
    
    input_file = "Matthew_Vaishnav_Harvard_Style_Resume.md"
    output_file = "Matthew_Vaishnav_Harvard_Style_Resume.pdf"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found!")
        return False
    
    # Pandoc command with professional formatting
    cmd = [
        "pandoc",
        input_file,
        "-o", output_file,
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=0.75in",
        "-V", "fontsize=11pt",
        "-V", "mainfont=Times New Roman",
        "-V", "linestretch=1.1",
        "--highlight-style=tango"
    ]
    
    try:
        print(f"Converting {input_file} to {output_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Successfully created {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating PDF: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    
    except FileNotFoundError:
        print("❌ Error: pandoc not found!")
        print("Install pandoc: https://pandoc.org/installing.html")
        print("Windows: choco install pandoc")
        print("Or download from: https://github.com/jgm/pandoc/releases")
        return False

if __name__ == "__main__":
    success = create_pdf()
    sys.exit(0 if success else 1)