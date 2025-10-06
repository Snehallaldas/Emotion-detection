from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

# Initialize FastAPI
app = FastAPI(title="Emotion Detection Web App")

# Load model and vectorizer
vectorizer, model = joblib.load("model/sentiment_model.pkl")

# Connect templates folder
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    transformed = vectorizer.transform([text])
    pred = model.predict(transformed)[0]
    label = "😠 Angry" if pred == 0 else "😊 Happy"
    return templates.TemplateResponse("index.html", {"request": request, "prediction": label, "text": text})
