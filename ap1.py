from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
from gliner import GLiNER
import cv2
import re
import json
from PIL import Image, ExifTags
import numpy as np


# Initialisation des modèles
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
ocr = PaddleOCR(
    use_angle_cls=True,        
    image_orientation=True,    
    lang='en', 
    det=True,
    rec=True
)

######################################################################################################################################################
# #################################################  Les Fonctions    ################################################################################
# #################################################################################################################################################### 

def resize_image(image_path, width=800):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = image.shape[:2] 

    scale = width / w
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return resized_image, scale


def preprocessing(line):

    parts = line.split(":", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return line

# Calculer la hauteur ou largeur minimale du texte pour le filtrer
def is_too_small(bbox, min_size=10):

    width = abs(bbox[1][0] - bbox[0][0])
    height = abs(bbox[2][1] - bbox[1][1])
    return width < min_size or height < min_size

# Regex pour certaines entités fixes
def match_regex(line):
    patterns = {
        "Email": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})|(\bmail\b)",
        "Phone": r"^[\sTtEeLl.:,)(-/+0-9]+$",
        "Website": r"https?://[^\s]+|www\.[^\s]+",
        "Street Address": r"\d{1,5}\s[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Library|Building|Bldg|Way|Lane|Ln|Drive|Dr)\b",
    }

    for label, pattern in patterns.items():
        if re.search(pattern, line, re.IGNORECASE):
            return label
    return None



def clean_and_merge(data):
    address_parts = {"City": "", "Street Address": "", "Postal Code": ""}
    result = []

    for item in data:
        label = item["label"]
        text = item["text"]

        if label == "Unknown":
            continue   

        if (label == "Phone") and (len(text) > 15):
            parts = text.split()
            for part in parts:
                result.append({"label": "Phone", "text": part})
            continue    

        if label in address_parts:
            address_parts[label] = text
        else:
            result.append(item) 

    full_address = ", ".join(
        part for part in [address_parts["Street Address"], address_parts["City"], address_parts["Postal Code"]] if part
    )
    if full_address:
        result.append({"label": "Address", "text": full_address})

    return result

def clean_phone_number(s):
    # Nettoyer le numéro en ne gardant que chiffres et +
    s = re.sub(r"[^\d+]", "", s)

    if len(s) > 15 and len(s) % 2 == 0:
        mid = len(s) // 2
        s1 = s[:mid]
        s2 = s[mid:]
        return s1, s2
    else:
        return s


# Paramètres
image_path = 'test/010.jpg'
doc = ""

labels = [
    "Full Name",       
    "Job Title",       
    "Organization",    
    "Street Address",  
    "City",            
    "Postal Code",     
    "Country",
    "Phone"         
]

# Redimensionner l'image
resized_img, scale = resize_image(image_path)

# Lancer OCR sur l’image redimensionnée
results = ocr.ocr(resized_img)


boxes, texts, scores = [], [], []

for line in results[0]:
    bbox = line[0]
    text = line[1][0]
    confidence = line[1][1]

    if is_too_small(bbox):
        continue  # Ignorer les petits textes

    doc += text + "\n"
    boxes.append(bbox)
    texts.append(text)
    scores.append(confidence)


print("--------------------------------------------------------------------------------------------------------------------------------")
print(doc)
print("-------------------------------------------------------NER----------------------------------------------------------------------")



# Lister les résultats ligne par ligne
labeled_lines = []

for line in texts:
    label = match_regex(line)
    line = preprocessing(line)

    if label:
        if label == "Phone":
            result = clean_phone_number(line)
            if isinstance(result, tuple):
                # si divisé en 2 parties
                line = " ".join(result)
            else:
                line = result

            if len(line) < 8:
                label = "Unknown"

        labeled_lines.append((line, label))

    else:
        entities = model.predict_entities(line, labels, threshold=0.4)
        if entities:
            entities = sorted(entities, key=lambda x: x["score"], reverse=True)
            best_entity = entities[0]

            if best_entity["label"] == "Phone":
                result = clean_phone_number(line)
                if isinstance(result, tuple):
                    line = " ".join(result)
                else:
                    line = result

                if len(line) < 8:
                    best_entity = {"label": "Unknown", "score": 0}

            if best_entity["label"] == "Full Name":
                if len(line.split()) == 1:
                    best_entity = {"label": "Organization", "score": 0}

                if any(c.isdigit() for c in line):
                    for entity in entities[1:]:
                        print(entity["label"])
                        if entity["label"] != "Full Name":
                            best_entity = entity
                            break
                    else:
                        best_entity = {"label": "Unknown", "score": 0}

            labeled_lines.append((line, best_entity["label"]))

        else:
            labeled_lines.append((line, "Unknown"))


# Concaténer les lignes consécutives avec la même étiquette
merged_results = []
prev_label = None
current_text = ""

for line, label in labeled_lines:
    if label == prev_label and label!="Phone" and label!="Full Name":
        current_text += " " + line
    else:
        if prev_label is not None:
            merged_results.append({"label": prev_label, "text": current_text.strip()})
        current_text = line
        prev_label = label

# Ajouter le dernier bloc
if current_text and prev_label:
    merged_results.append({"label": prev_label, "text": current_text.strip()})

merged_results=clean_and_merge(merged_results)

print (merged_results)

# Sauvegarder dans un fichier JSON
output_path = "labeled_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_results, f, indent=4, ensure_ascii=False)

print(f"Résultats enregistrés dans {output_path}")