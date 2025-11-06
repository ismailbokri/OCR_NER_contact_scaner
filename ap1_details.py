from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
from gliner import GLiNER
import cv2
import re
import json
from PIL import Image, ExifTags
import numpy as np
import psutil
import time
import os
import threading

process = psutil.Process()
freq = psutil.cpu_freq()
cpu_freq_ghz = freq.current / 1000  
num_cores = psutil.cpu_count()
print(f"Fréquence actuelle d'un cœur : {cpu_freq_ghz:.2f} GHz")
print(f"Nombre total de cœurs        : {num_cores}")

# pour stocker les valeurs mesurées pendant l'exécution
cpu_samples = []
mem_samples = []
running = True

def monitor():
    """ Fonction qui échantillonne CPU et mémoire pendant que le code tourne """
    while running:
        cpu_percent = psutil.cpu_percent(interval=0.1)  # court intervalle
        memory = process.memory_info().rss / 1024 / 1024
        cpu_load_ghz = num_cores * cpu_freq_ghz * (cpu_percent / 100)

        cpu_samples.append(cpu_load_ghz)
        mem_samples.append(memory)

# démarrer le monitoring dans un thread à part
monitor_thread = threading.Thread(target=monitor)
monitor_thread.start()

start = time.time()






###############################################################################################################################################


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




# Calculer la hauteur ou largeur minimale du texte pour le filtrer
def is_too_small(bbox, min_size=10):

    width = abs(bbox[1][0] - bbox[0][0])
    height = abs(bbox[2][1] - bbox[1][1])
    return width < min_size or height < min_size

# Regex pour certaines entités fixes
def match_regex(line):
    patterns = {
        "Email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "Phone": r"^[\sTtEeLl.,)(-/+0-9]+$",
        "Website": r"https?://[^\s]+|www\.[^\s]+",
        "Street Address": r"\d{1,5}\s[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Library|Building|Bldg|Way|Lane|Ln|Drive|Dr)\b",
    }

    for label, pattern in patterns.items():
        if re.search(pattern, line, re.IGNORECASE):
            return label
    return None

def clean_phone_number(s):
    return re.sub(r"[^\d+]", "", s)

def clean_and_merge(data):
    # Dictionnaire pour récupérer city, street, postal
    address_parts = {"City": "", "Street Address": "", "Postal Code": ""}
    result = []

    for item in data:
        label = item["label"]
        text = item["text"]

        if label == "Unknown":
            continue  # on ignore Unknown

        if label in address_parts:
            address_parts[label] = text
        else:
            result.append(item)  # on garde les autres tels quels

    # On crée le champ address fusionné
    full_address = ", ".join(
        part for part in [address_parts["Street Address"], address_parts["City"], address_parts["Postal Code"]] if part
    )
    if full_address:
        result.append({"label": "Address", "text": full_address})

    return result




# Paramètres
image_path = '055.jpg'
doc = ""

labels = [
    "Full Name",       
    "Job Title",       
    "Organization",    
    "Street Address",  
    "City",            
    "Postal Code",     
    "Country"         
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
    
    if label:
        if label=="Phone":
            line=clean_phone_number(line)
            if len(line) < 8:
                label="Unknown"
        labeled_lines.append((line, label))
    else:
        entities = model.predict_entities(line, labels, threshold=0.2)
        if entities:
            entities = sorted(entities, key=lambda x: x["score"], reverse=True)
            best_entity = entities[0]

            if best_entity["label"] == "Full Name" and any(c.isdigit() for c in line):
                
                for entity in entities[1:]:
                    if entity["label"] != "Full Name":
                        best_entity = entity
                        break
                    else:
                        best_entity = "Unknown"
            
            labeled_lines.append((line, best_entity["label"]))

        else:
            labeled_lines.append((line, "Unknown"))



# Concaténer les lignes consécutives avec la même étiquette
merged_results = []
prev_label = None
current_text = ""

for line, label in labeled_lines:
    if label == prev_label and label!="Phone":
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





###########################################################################################################################################################
end = time.time()

# arrêter le monitoring
running = False
monitor_thread.join()

# calculer moyennes et pics
avg_cpu_load = np.mean(cpu_samples)
max_cpu_load = np.max(cpu_samples)
avg_mem = np.mean(mem_samples)
max_mem = np.max(mem_samples)

print(f"\n--- Résultats ---")
print(f"Durée totale       : {end - start:.2f} s")
print(f"Charge CPU moyenne : {avg_cpu_load:.2f} GHz cumulés")
print(f"Charge CPU max     : {max_cpu_load:.2f} GHz cumulés")
print(f"RAM moyenne        : {avg_mem:.2f} MB")
print(f"RAM max            : {max_mem:.2f} MB")