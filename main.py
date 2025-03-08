import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
import cv2
import spacy
import re
import numpy as np
from PIL import Image
import easyocr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Ensure API Key is properly loaded
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not set in environment variables")


@app.get("/")
def read_root():
    return {"message": "Welcome to Electrothon API!"}

# Load Spacy Model (Download if not available)
spacy_model = "en_core_web_sm"
try:
    nlp = spacy.load(spacy_model)
except OSError:
    spacy.cli.download(spacy_model)
    nlp = spacy.load(spacy_model)

# Initialize EasyOCR Reader
reader = easyocr.Reader(["en"], gpu=False)

# FDA Allergen Data
FDA_ALLERGENS = {
    "milk": ["milk", "lactose", "butter", "cheese", "cream"],
    "eggs": ["eggs", "egg whites", "egg yolks", "albumin"],
    "fish": ["fish", "salmon", "tuna", "cod"],
    "shellfish": ["shrimp", "lobster", "crab", "shellfish"],
    "tree nuts": ["almonds", "cashews", "hazelnuts", "pistachios"],
    "peanuts": ["peanuts", "peanut butter"],
    "wheat": ["wheat", "gluten", "flour", "wheat gluten"],
    "soy": ["soy", "soybean", "tofu", "soya", "soy protein"],
    "sesame": ["sesame", "tahini"]
}

COMMON_INGREDIENTS = set()
for allergen_list in FDA_ALLERGENS.values():
    COMMON_INGREDIENTS.update(allergen_list)
COMMON_INGREDIENTS.update(["sugar", "salt", "flour", "oil", "butter", "yeast", "cheese", "gluten"])


def authenticate(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=20)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return gray


def extract_text(image: np.ndarray) -> str:
    processed_img = preprocess_image(image)
    extracted_text = " ".join(reader.readtext(processed_img, detail=0))
    return extracted_text


def clean_text(text: str) -> str:
    return re.sub(r'[^a-z\s-]', '', text.lower())


def extract_ingredients(text: str):
    words = set(text.split())
    return list(words.intersection(COMMON_INGREDIENTS))


def check_allergens(ingredients):
    allergens_found = [allergen for allergen, synonyms in FDA_ALLERGENS.items() if any(ingredient in synonyms for ingredient in ingredients)]
    return allergens_found


@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...), api_key: str = Depends(authenticate)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    extracted_text = extract_text(image)
    cleaned_text = clean_text(extracted_text)
    ingredients = extract_ingredients(cleaned_text)
    allergens = check_allergens(ingredients)

    return {
        "ingredients": ingredients,
        "allergens": allergens
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
