# PDF Banner Processor

## Overview
A Flask web application that processes PDF files by detecting and removing competitor banner images using template matching, and adding custom banners at the bottom of each page.

## Project Structure
```
.
├── app.py                 # Main Flask application with routes
├── pdf_processor.py       # PDF processing logic with OpenCV
├── templates/
│   ├── index.html         # PDF upload page
│   └── banners.html       # Banner management page
├── static/
│   └── style.css          # Basic styling
├── uploads/
│   ├── pdfs/              # Uploaded PDF files
│   └── banners/
│       ├── competitor/    # Competitor banner templates for detection
│       └── custom/        # Custom banners to add to PDFs
└── processed/             # Output processed PDF files
```

## Features
1. **PDF Upload**: Upload PDF files for processing
2. **Banner Management**: Upload competitor banners (to detect/remove) and custom banners (to add)
3. **Template Matching**: Uses OpenCV to detect competitor banners in PDF pages
4. **Banner Insertion**: Adds custom banner at the bottom of each processed page

## Dependencies
- Flask (web framework)
- OpenCV (opencv-python-headless) for template matching
- PyPDF2 for PDF manipulation
- Pillow (PIL) for image processing
- pdf2image for PDF to image conversion
- reportlab for PDF generation
- poppler-utils (system dependency for pdf2image)

## Running the App
```bash
python app.py
```
The app runs on port 5000.

## Usage
1. First, upload banner templates via the "Manage Banners" page
   - Upload competitor banners (images to detect and remove)
   - Upload your custom banner (image to add at bottom)
2. Go to "Process PDF" page and upload a PDF file
3. The processed PDF will be automatically downloaded

## Configuration
- `match_threshold` in `pdf_processor.py`: Controls template matching sensitivity (default: 0.8)
- Maximum file size: 50MB (configurable in `app.py`)
