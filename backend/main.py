from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import detect

app = FastAPI(title="AI Content Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)

@app.get("/")
async def root():
    return {"message": "AI Content Detector API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
