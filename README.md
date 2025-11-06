# 📇 Business Card Scanner – Automatic Contact Extraction using OCR & NER

## 🧠 Overview

This project presents an **automatic business card scanning system** that simplifies the process of adding new contacts from physical cards.  
The system extracts and classifies text information from business card images using a hybrid **OCR + NER** pipeline.

---

## 🚀 How It Works

1. **Image Input**  
   The user provides an image of a business card.

2. **Text Extraction (OCR Stage)**  
   The image is passed to a pre-trained **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** model, which extracts all visible text from the card with high accuracy.

3. **Entity Recognition (NER Stage)**  
   The extracted text is then processed by a **Named Entity Recognition (NER)** model to identify and classify key information such as:
   - 👤 **Name / Surname**
   - 🏢 **Company / Job Title**
   - 📧 **Email**
   - 📞 **Phone Number**
   - 📍 **Address**

4. **Hybrid Classification Approach**  
   The NER stage uses a **hybrid method** that combines:
   - **Regular expressions (RegEx)** for pattern-based entity detection (emails, phone numbers, etc.).
   - **[Gliner](https://github.com/urchade/GLiNER)** model — a transformer-based NER architecture that allows **custom label definition without retraining**, providing flexibility and adaptability across languages and formats.

---

## 🧩 Technologies Used

| Component | Technology |
|------------|-------------|
| OCR Engine | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| NER Model | [Gliner](https://github.com/urchade/GLiNER) |
| Text Preprocessing | Python RegEx |
| Frameworks | Python, PyTorch |
| Output Format | JSON (structured contact information) |

---
