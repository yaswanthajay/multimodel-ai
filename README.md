### 📦 Dependencies

- Python 3.x
- pytesseract
- pillow (PIL)
- Tesseract OCR Engine

### ⚙️ Install Tesseract on Windows

1. Download from: https://github.com/tesseract-ocr/tesseract
2. Install to: `C:\Program Files\Tesseract-OCR\`
3. Add to your System PATH:

### 🛠 Tesseract Setup (Windows)

- Install from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

In your Python code:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
