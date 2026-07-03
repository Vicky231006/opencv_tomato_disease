# app.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

# Ensure the server uses the correct hardware topology
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the main engine class from our pipeline module file
from pipeline_module import CropDiseasePipeline

app = FastAPI(
    title="AgTech Early Disease Detection API",
    description="Production ML inference server for processing crop leaf pathologies.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime engine initialization instance (loads weight files into memory once)
pipeline_engine = CropDiseasePipeline(
    verifier_path="leaf_verifier.pth", 
    classifier_path="tomato_classifier.pth"
)

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "device_target": str(device)}

@app.post("/api/v1/diagnose")
async def diagnose_leaf(file: UploadFile = File(...)):
    # Validate payload extension formats
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload JPG, JPEG, or PNG images.")
    
    # Secure staging folder path allocation
    staging_dir = "./server_staging"
    os.makedirs(staging_dir, exist_ok=True)
    temp_file_path = os.path.join(staging_dir, file.filename)
    
    try:
        # Stream incoming upload payload into disk buffer space
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # ⚡ UNCOMMENTED & ACTIVATED: Execute real modular pipeline evaluation flow
        pipeline_report = pipeline_engine.run_pipeline(temp_file_path)
        
        if pipeline_report["status"] == "Rejected":
            raise HTTPException(status_code=422, detail=pipeline_report.get("reason", "Image rejected by quality validation checkpoints."))
            
        # Dynamically build and format response matching data out of the live classification run
        return {
            "success": True,
            "data": {
                "crop": pipeline_report["crop"],
                "condition": "PATHOGEN DETECTED" if not pipeline_report["is_healthy"] else "HEALTHY",
                "diagnosed_class": pipeline_report["disease"].replace("_", " "),
                "severity_percentage": pipeline_report["severity"],
                "treatment_protocol": pipeline_report["treatment"]["Remedy"]
            }
        }
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Inference Error: {str(e)}")
        
    finally:
        # Always purge staging file to eliminate storage leaks
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)