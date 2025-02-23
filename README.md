# Object-detection-with-colour-and-text-recognition
Using YOLOv9-seg, rule-based color classifier, and easyOCR


# Overview

This model perform an object recognition. It also try to indentify the color of detected object and overlaying text if there's any. AI model used in the project:

- [YOLOv9 tiny and YOLOv9c-seg](https://docs.ultralytics.com/models/yolov9)
- [HSV rule-based color classifier (self-made)](https://github.com/MRX760/Object-detection-with-colour-and-text-recognition/blob/main/utils/color_classifier.py)
- [easyOCR](https://github.com/JaidedAI/EasyOCR)

# Requirements

Python packages:
- ultralytics
- pytorch (for faster computation)
- easyocr
- scikit-learn
- pyttsx3

# How to use 
Clone repository:
```bash
git clone https://github.com/MRX760/Personal-chatbot.git
```

Install pre-requisites:
```bash 
cd Object-detection-with-colour-and-text-recognition #navigate to cloned repo folder
pip install -r requirements.txt #install dependencies
```
> **Note:** Install supported version of pytorch on your device to ensure faster computation

Run the main program:
```bash
python main_parallel.py
```
# General Inference flow 

In parallel: 
![parallel inference flow image](https://github.com/MRX760/Object-detection-with-colour-and-text-recognition/blob/main/assets/parallel-flow.png)

Detailed flow (not parallel process):
![Detailed flow image](https://github.com/MRX760/Object-detection-with-colour-and-text-recognition/blob/main/assets/detailed-flow.png)
> **Note:** actually, it's still in Bahasa. I'll update it to english once the repo are quite famous 😆