import os, torch, shutil
from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, GPT2Tokenizer
from model import StableSalientCaptioner
from Predict import Caption
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app = FastAPI()
os.makedirs("uploads", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_dir = "B:/SatCap/trained_model-20251107T123900Z-1-001/trained_model"

print("Loading model and tokenizer...")


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = tokenizer.eos_token

model = StableSalientCaptioner(tokenizer)
model_path = os.path.join(model_dir, "final_salient_caption_model.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(" Model ready!")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        cap = Caption(file_path, model, processor, tokenizer, device)
        caption = cap.generate_quality_caption()
        return {"filename": file.filename, "caption": caption}
    except Exception as e:
        return {"error": str(e)}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")
