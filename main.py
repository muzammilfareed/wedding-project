from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import zipfile
import time
import shutil
import uvicorn
import cv2
from insightface.app import FaceAnalysis
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List
import requests
import json
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


app_faces = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app_faces.prepare(ctx_id=0, det_thresh=0.75, det_size=(640, 640))  # ctx_id=0 -> use GPU
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
SIMILARITY_THRESHOLD = 0.5
# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


def flatten_image_folder(source_dir: str):
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
    moved_count = 0

    for dirpath, _, filenames in os.walk(source_dir, topdown=False):
        for file in filenames:
            if file.lower().endswith(valid_exts):
                src_path = os.path.join(dirpath, file)
                dst_path = os.path.join(source_dir, file)

                # Handle duplicate names
                if os.path.exists(dst_path):
                    name, ext = os.path.splitext(file)
                    count = 1
                    while True:
                        new_name = f"{name}_{count}{ext}"
                        new_dst = os.path.join(source_dir, new_name)
                        if not os.path.exists(new_dst):
                            dst_path = new_dst
                            break
                        count += 1

                shutil.move(src_path, dst_path)
                moved_count += 1

    for dirpath, dirnames, filenames in os.walk(source_dir, topdown=False):
        if dirpath != source_dir and not os.listdir(dirpath):
            os.rmdir(dirpath)


def save_image_from_url(url, save_dir):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        filename = url.split("/")[-1].split("?")[0]
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        return file_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None



import json

def process_images_parallel(image_urls_list, target_folder, target_folder_embs):
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(target_folder_embs, exist_ok=True)

    status_file = os.path.join(target_folder, "status.json")
    # If status.json exists, update total_images, else create
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
        status_data["total_images"] += len(image_urls_list)
    else:
        status_data = {
            "total_images": len(image_urls_list),
            "processed_images": 0,
            "status": "processing"
        }
    with open(status_file, "w") as f:
        json.dump(status_data, f)

    # --- Step 1: Download images ---
    def download(url):
        return save_image_from_url(url, target_folder)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download, url) for url in image_urls_list]
        saved_files = [f.result() for f in as_completed(futures) if f.result()]

    flatten_image_folder(target_folder)

    # --- Step 2: Extract faces ---
    imgs_list = os.listdir(target_folder)

    def extract_face(img_file):
        img_path = os.path.join(target_folder, img_file)

        # --- Always increment processed count at the start ---
        try:
            with open(status_file, "r+") as sf:
                data = json.load(sf)
                data["processed_images"] += 1
                if data["processed_images"] > data["total_images"]:
                    data["processed_images"] = data["total_images"]
                sf.seek(0)
                json.dump(data, sf)
                sf.truncate()
        except Exception as e:
            print("Error updating status for:", img_file, e)

        # --- Try to process the image ---
        try:
            img_np = cv2.imread(img_path)
            if img_np is None:
                raise ValueError("Failed to read image")

            faces = app_faces.get(img_np)
            embeddings = [face.embedding for face in faces if hasattr(face, "embedding")]

            if embeddings:
                emb_name = os.path.splitext(img_file)[0] + ".pkl"
                emb_path = os.path.join(target_folder_embs, emb_name)
                with open(emb_path, "wb") as f:
                    pickle.dump(embeddings, f)

            return True
        except Exception as e:
            print("Error processing:", img_file, e)
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_face, img) for img in imgs_list]
        for f in as_completed(futures):
            f.result()  # wait for completion

    # Mark complete
    with open(status_file, "r+") as f:
        data = json.load(f)
        data["status"] = "completed"
        f.seek(0)
        json.dump(data, f)
        f.truncate()




from fastapi import BackgroundTasks

@app.post("/upload_urls")
async def upload_urls(
    background_tasks: BackgroundTasks,
    image_urls: str = Form(...),
    wedding_name: str = Form(...),
    wedding_folder_id: Optional[str] = Form(None)
):
    timestamp = int(time.time())
    folder_name = f"{wedding_name}_{timestamp}"
    folder_name_embs = f"{wedding_name}_{timestamp}_embeddings"
    if wedding_folder_id:
        folder_name = wedding_folder_id
        folder_name_embs = f"{folder_name}_embeddings"

    target_folder = os.path.join(UPLOAD_DIR, folder_name)
    target_folder_embs = os.path.join(UPLOAD_DIR, folder_name_embs)

    image_urls_list = json.loads(image_urls)

    background_tasks.add_task(process_images_parallel, image_urls_list, target_folder, target_folder_embs)

    return JSONResponse({
        "message": "Upload started ✅. Face extraction is processing in the background.",
        "wedding_folder_id": folder_name
    })


@app.get("/check_status/{wedding_folder_id}")
async def check_status(wedding_folder_id: str):
    folder_path = os.path.join(UPLOAD_DIR, wedding_folder_id)
    status_file = os.path.join(folder_path, "status.json")

    if not os.path.exists(status_file):
        return JSONResponse({"message": "No task found for this folder", "status": "not_found"})

    with open(status_file, "r+") as f:
        data = json.load(f)
        # Only mark as completed when all images are processed
        if data["processed_images"] >= data["total_images"]:
            data["status"] = "completed"
        else:
            data["status"] = "processing"
        # Save the updated status
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    return JSONResponse(data)


# @app.post("/upload_urls")
# async def upload_urls(
#         image_urls: str = Form(...),
#         wedding_name: str = Form(...),
#         wedding_folder_id: Optional[str] = Form(None)
# ):
#     timestamp = int(time.time())
#     print("wedding_folder_id:", wedding_folder_id)
#     folder_name = f"{wedding_name}_{timestamp}"
#     folder_name_embs = f"{wedding_name}_{timestamp}_embeddings"
#     if wedding_folder_id:
#         folder_name = wedding_folder_id
#         folder_name_embs = f"{folder_name}_embeddings"
#
#     target_folder = os.path.join(UPLOAD_DIR, folder_name)
#     target_folder_embs = os.path.join(UPLOAD_DIR, folder_name_embs)
#     os.makedirs(target_folder, exist_ok=True)
#     os.makedirs(target_folder_embs, exist_ok=True)
#
#     # ✅ Download images from URLs
#     image_urls = json.loads(image_urls)
#     print(len(image_urls))
#     saved_files = []
#     # image_urls = json.loads(image_urls)
#     for url in image_urls:
#         print('url: ', url)
#         saved = save_image_from_url(url, target_folder)
#         print("save:", saved)
#         if saved:
#             saved_files.append(saved)
#
#     # ✅ Flatten or further process
#     flatten_image_folder(target_folder)
#
#     imgs_list = os.listdir(target_folder)
#     if len(imgs_list) > 0:
#         status = False
#         for img in imgs_list:
#             try:
#                 img_path = os.path.join(target_folder, img)
#                 img_np = cv2.imread(img_path)
#                 faces = app_faces.get(img_np)
#                 known_faces = []
#                 for face in faces:
#                     emb = face.embedding
#                     known_faces.append(emb)
#                 img_name = os.path.splitext(img)[0]
#                 emb_name = img_name + ".pkl"
#                 emb_path = os.path.join(target_folder_embs, emb_name)
#                 if len(known_faces) > 0:
#                     with open(emb_path, "wb") as f:
#                         pickle.dump(known_faces, f)
#                     status = True
#             except:
#                 pass
#         if status is True:
#             return JSONResponse({
#                 "message": "Upload & Face extractions successful ✅",
#                 "wedding_folder_id": folder_name
#             })
#         else:
#             shutil.rmtree(target_folder)
#             shutil.rmtree(target_folder_embs)
#             return JSONResponse({
#                 "message": "Somthing went wrong. Either No faces detected or folder is empty.",
#                 "wedding_folder_id": None
#             })
#     else:
#         return JSONResponse({
#             "message": "Somthing went wrong. Either No faces detected or folder is empty.",
#             "wedding_folder_id": None
#         })

from concurrent.futures import ThreadPoolExecutor, as_completed

@app.post("/find_person")
async def find_face_fast(
        input_img: UploadFile,
        wedding_folder_id: str = Form(...)
):
    # timestamp = int(time.time())

    # Save input image
    img_path = os.path.join(UPLOAD_DIR, input_img.filename)
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(input_img.file, buffer)

    folder_name = f"{wedding_folder_id}"
    folder_name_embs = f"{wedding_folder_id}_embeddings"
    target_folder = os.path.join(UPLOAD_DIR, folder_name)
    target_folder_embs = os.path.join(UPLOAD_DIR, folder_name_embs)

    if not os.path.exists(target_folder) or not os.path.exists(target_folder_embs):
        return JSONResponse({
            "message": "Wedding folder not found for given ID.",
            "match_list": []
        })

    # Extract target face embedding
    target_img_np = cv2.imread(img_path)
    if target_img_np is None:
        return JSONResponse({"message": "Invalid image format", "match_list": []})

    faces = app_faces.get(target_img_np)
    if not faces:
        return JSONResponse({"message": "No face detected in input image", "match_list": []})

    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    target_embedding = largest_face.embedding.reshape(1, -1)

    # === Step 1: Load all embeddings into memory efficiently ===
    embs_list = os.listdir(target_folder_embs)
    loaded_data = []
    for emb_file in embs_list:
        emb_path = os.path.join(target_folder_embs, emb_file)
        try:
            with open(emb_path, "rb") as f:
                embeddings = pickle.load(f)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            else:
                embeddings = np.array([embeddings])
            loaded_data.append((emb_file, embeddings))
        except Exception as e:
            print(f"Failed to load {emb_file}: {e}")

    # === Step 2: Compute similarities in parallel batches ===
    batch_size = 256  # tune for performance depending on CPU cores
    matches = []

    def process_batch(batch_data):
        local_matches = []
        for emb_name, emb_array in batch_data:
            sims = cosine_similarity(target_embedding, emb_array)[0]
            best_sim = np.max(sims)
            if best_sim >= SIMILARITY_THRESHOLD:
                img_base = os.path.splitext(emb_name)[0]
                img_candidates = [
                    os.path.join(target_folder, file)
                    for file in os.listdir(target_folder)
                    if img_base in file
                ]
                for img_path in img_candidates:
                    local_matches.append(f"{img_path}")
        return local_matches

    # Run multi-threaded similarity search
    num_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(loaded_data), batch_size):
            batch = loaded_data[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch))

        for future in as_completed(futures):
            matches.extend(future.result())

    if matches:
        # matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        return JSONResponse({
            "status": True,
            "message": "Matches found",
            "match_count": len(matches),
            "matches": matches  # return top 50 matches for performance
        })
    else:
        return JSONResponse({
            "status": False,
            "message": "No similar faces found.",
            "match_count": 0,
            "matches": []
        })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=False)

