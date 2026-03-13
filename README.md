# BackPack Automation Scripts

## PDF employment-doc redaction workflow

This repository now includes two Python scripts to help prepare paystubs and W-2 PDFs for HireRight employment verification.

### 1) Extract all PDF text

```bash
python pdf_text_scraper.py /path/to/paystub.pdf /path/to/w2.pdf
```

By default, extracted text files are written to `./extracted_text/`.

### 2) Redact compensation/pay-history content in PDFs

```bash
python pdf_redactor.py /path/to/paystub.pdf /path/to/w2.pdf
```

By default, redacted PDFs are written to `./redacted_pdfs/` using the suffix `_redacted.pdf`.

## Install

```bash
python -m pip install -r requirements.txt
```

> Note: These scripts work best on text-based PDFs. If your PDF is a scanned image, run OCR first and then redact.
