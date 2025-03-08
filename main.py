import uvicorn
import pickle
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
import cv2
import pytesseract
import spacy
import re
import numpy as np
from PIL import Image
import easyocr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Set your API key
API_KEY = os.getenv("API_KEY", "my_secure_api_key")  # Default key if not set

@app.get("/")
def read_root():
    return {"message": "Welcome to Electrothon API!"}


nlp = spacy.load("en_core_web_sm")

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
    pil_image = Image.fromarray(processed_img)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(pil_image, config=custom_config)

    reader = easyocr.Reader(["en"], gpu=False)
    easy_text = " ".join(reader.readtext(image, detail=0))

    return text + " " + easy_text


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
    uvicorn.run(app, host="0.0.0.0", port=8000)
