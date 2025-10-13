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
app_faces = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app_faces.prepare(ctx_id=0, det_thresh=0.75, det_size=(640, 640))  # ctx_id=0 -> use GPU

app = FastAPI()

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

# @app.post("/upload")
# async def upload_zip(
#     data_zip: UploadFile,
#     wedding_name: str = Form(...)
# ):
#     timestamp = int(time.time())
#     folder_name = f"{wedding_name}_{timestamp}"
#     folder_name_embs = f"{wedding_name}_{timestamp}_embeddings"
#     target_folder = os.path.join(UPLOAD_DIR, folder_name)
#     target_folder_embs = os.path.join(UPLOAD_DIR, folder_name_embs)
#     os.makedirs(target_folder, exist_ok=True)
#     os.makedirs(target_folder_embs, exist_ok=True)

#     zip_path = os.path.join(target_folder, data_zip.filename)
#     with open(zip_path, "wb") as buffer:
#         shutil.copyfileobj(data_zip.file, buffer)

#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(target_folder)

#     os.remove(zip_path)

#     flatten_image_folder(target_folder)
@app.post("/upload_urls")
async def upload_urls(
    image_urls: str = Form(...), 
    wedding_name: str = Form(...)
):
    timestamp = int(time.time())
    folder_name = f"{wedding_name}_{timestamp}"
    folder_name_embs = f"{wedding_name}_{timestamp}_embeddings"
    
    target_folder = os.path.join(UPLOAD_DIR, folder_name)
    target_folder_embs = os.path.join(UPLOAD_DIR, folder_name_embs)
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(target_folder_embs, exist_ok=True)

    # ✅ Download images from URLs
    image_urls = json.loads(image_urls)
    print(len(image_urls))
    saved_files = []
    # image_urls = json.loads(image_urls)
    for url in image_urls:
        print('url: ',url)
        saved = save_image_from_url(url, target_folder)
        print("save:",saved)
        if saved:
            saved_files.append(saved)

    # ✅ Flatten or further process
    flatten_image_folder(target_folder)

    imgs_list = os.listdir(target_folder)
    if len(imgs_list) > 0:
        status = False
        for img in imgs_list:
            try:
                img_path = os.path.join(target_folder, img)
                img_np = cv2.imread(img_path)
                faces = app_faces.get(img_np)
                known_faces = []
                for face in faces:
                    emb = face.embedding
                    known_faces.append(emb)
                img_name = os.path.splitext(img)[0]
                emb_name = img_name + ".pkl"
                emb_path = os.path.join(target_folder_embs, emb_name)
                if len(known_faces) > 0:
                    with open(emb_path, "wb") as f:
                        pickle.dump(known_faces, f)
                    status = True
            except:
                pass
        if status is True:
            return JSONResponse({
                "message": "Upload & Face extractions successful ✅",
                "wedding_folder_id": folder_name
            })
        else:
            shutil.rmtree(target_folder)
            shutil.rmtree(target_folder_embs)
            return JSONResponse({
                "message": "Somthing went wrong. Either No faces detected or folder is empty.",
                "wedding_folder_id": None
            })
    else:
        return JSONResponse({
            "message": "Somthing went wrong. Either No faces detected or folder is empty.",
            "wedding_folder_id": None
        })


@app.post("/find_person")
async def find_face(
    input_img: UploadFile,
    wedding_folder_id: str = Form(...)
):
    timestamp = int(time.time())

    img_path = os.path.join(UPLOAD_DIR, input_img.filename)
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(input_img.file, buffer)

    folder_name = f"{wedding_folder_id}"
    folder_name_embs = f"{wedding_folder_id}_embeddings"
    target_folder = os.path.join(UPLOAD_DIR, folder_name)
    target_folder_embs = os.path.join(UPLOAD_DIR, folder_name_embs)

    target_img_name = os.path.splitext(input_img.filename)[0]
    target_img_dir = os.path.join(UPLOAD_DIR, target_img_name)
    os.makedirs(target_img_dir, exist_ok=True)

    target_img_np = cv2.imread(img_path)
    faces = app_faces.get(target_img_np)
    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    target_embedding = largest_face.embedding
    find_match = []
    image_name = 1

    if os.path.exists(target_folder) and os.path.exists(target_folder_embs):
        wedding_imgs = os.listdir(target_folder)
        imgs_list = os.listdir(target_folder)
        embs_list = os.listdir(target_folder_embs)
        for emb in embs_list:
            emb_name = os.path.splitext(emb)[0]
            emb_path = os.path.join(target_folder_embs, emb)
            with open(emb_path, "rb") as f:
                loaded_embeddings = pickle.load(f)
            sims = [cosine_similarity([target_embedding], [f])[0][0] for f in loaded_embeddings]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]

            if best_sim >= SIMILARITY_THRESHOLD:
                wedding_image_path = ""
                for wed in wedding_imgs:
                    if emb_name in wed:
                        wedding_image_path = os.path.join(target_folder, wed)
                if wedding_image_path:
                    img_name = os.path.basename(wedding_image_path)
                    fresh_img = cv2.imread(wedding_image_path)
                    print(f" → Face Found! (sim={best_sim if 'best_sim' in locals() else 'N/A'})")
                    f_name = f'output_{image_name}.JPG'
                    image_name += 1
                    save_path = os.path.join(target_img_dir, f_name)
                    cv2.imwrite(save_path, fresh_img)
                    find_match.append(f'http://157.173.221.163:8003/{save_path}')
                else:
                    print(f" → Face Image not Found! (sim={best_sim if 'best_sim' in locals() else 'N/A'})")

            else:
                print(f" → Face not Found! (sim={best_sim if 'best_sim' in locals() else 'N/A'})")
        if len(os.listdir(target_img_dir)) > 0:
            # shutil.make_archive(target_img_dir, 'zip', target_img_dir)
            # shutil.rmtree(target_img_dir)
            return JSONResponse({
                "message": "Images found successfully successful",
                "match_list": find_match
            })
        else:
            shutil.rmtree(target_img_dir)
            return JSONResponse({
                "message": "No images found for target.",
                "output_file": None
            })
    else:
        return JSONResponse({
            "message": "Wedding folder not found for given ID.",
            "output_file": None
        })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=False)
