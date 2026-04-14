# README.md

# Smart OCR App 🔍

A modern **Streamlit-based OCR (Image to Text) application** powered by **PaddleOCR**, **OpenCV**, and **Pillow**.

This app extracts text from screenshots, scanned documents, photos, dark-background images, and multilingual content using smart preprocessing and deep learning OCR.

## Features

- 📤 Upload images (`png`, `jpg`, `jpeg`, `webp`, `bmp`, `tiff`, `gif`)
- 🌍 Multi-language OCR support:
  - English (`en`)
  - Hindi (`hi`)
  - Chinese (`ch`)
- 🧠 Smart image preprocessing:
  - Upscaling low-resolution images
  - Contrast enhancement (CLAHE)
  - Noise reduction
  - Binarization / thresholding
- 📊 OCR confidence score
- 📈 Stats dashboard (words, lines, time taken)
- 📥 Download extracted text as `.txt`
- 🎨 Modern cyber-style custom UI
- 🔒 Runs fully local (no API / no cloud)

## Tech Stack

- Python
- Streamlit
- PaddleOCR
- PaddlePaddle
- OpenCV
- NumPy
- Pillow

## Project Structure

```bash
ocr_app.py
requirements.txt
README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run App

```bash
streamlit run ocr_app.py
```

## How It Works

1. User uploads an image.
2. App creates multiple processed versions of the image.
3. PaddleOCR runs on each version.
4. Best OCR result is selected using confidence score.
5. Extracted text is shown with downloadable output.

## Best Use Cases

- Notes / study material extraction
- Screenshot text copy
- Document digitization
- Hindi / English OCR
- Scanned PDFs converted to text (via image pages)
- Low-quality image text recovery

## Author Notes

Designed as a fast, free, local OCR tool with better preprocessing than basic OCR apps.

---


