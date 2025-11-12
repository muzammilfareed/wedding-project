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


import firebase_admin
from firebase_admin import credentials, storage
import os
# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'storageBucket': "surajproductions-3f28b.firebasestorage.app"
    })

bucket = storage.bucket()

# bucket = storage.bucket()


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




def upload_embedding_on_firebase(folder_name,local_file_path):
    file_name = os.path.basename(local_file_path)
    destination_blob_path = f"{folder_name}/{file_name}"

    # Upload the actual file
    blob = bucket.blob(destination_blob_path)
    blob.upload_from_filename(local_file_path)

import json

def process_images_parallel(image_urls_list, target_folder, target_folder_embs):
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(target_folder_embs, exist_ok=True)

    status_file = os.path.join(target_folder, "status.json")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
        status_data["total_images"] += len(image_urls_list)
        if status_data.get("all_images") is None:
            status_data["all_images"] = []

        # Now safely extend
        status_data["all_images"].extend(image_urls_list)
        # status_data["all_images"] = status_data["all_images"] + image_urls_list

    else:
        status_data = {
            "total_images": len(image_urls_list),
            "processed_images": 0,
            "status": "processing",
            "all_images": image_urls_list
        }
    with open(status_file, "w") as f:
        json.dump(status_data, f)

    # --- Step 1: Download images with status increment ---
    def download(url):
        try:
            result = save_image_from_url(url, target_folder)
            return result
        finally:
            # Increment counter even if download fails
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
                print("Error updating status during download:", e)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download, url) for url in image_urls_list]
        saved_files = [f.result() for f in as_completed(futures)]

    flatten_image_folder(target_folder)

    # --- Step 2: Extract faces ---
    imgs_list = os.listdir(target_folder)

    def extract_face(img_file):
        img_path = os.path.join(target_folder, img_file)
        try:
            img_np = cv2.imread(img_path)
            if img_np is None:
                raise ValueError("Failed to read image")

            faces = app_faces.get(img_np)
            embeddings = [face.embedding for face in faces if hasattr(face, "embedding")]
            print("embeddings: ",embeddings)

            if embeddings:
                emb_name = os.path.splitext(img_file)[0] + ".pkl"
                emb_path = os.path.join(target_folder_embs, emb_name)
                with open(emb_path, "wb") as f:
                    pickle.dump(embeddings, f)

                # upload_embedding_on_firebase(target_folder, emb_path)
                # if os.path.exists(emb_path):
                #     os.remove(emb_path)
                #     print(f"{emb_path} has been deleted")
                # else:
                #     print(f"{emb_path} does not exist")

                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"{img_path} has been deleted")
                else:
                    print(f"{img_path} does not exist")




            return True
        except Exception as e:
            print("Error processing image:", img_file, e)
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_face, img) for img in imgs_list]
        for f in as_completed(futures):
            f.result()  # wait for completion

    # --- Mark complete ---
    with open(status_file, "r+") as f:
        data = json.load(f)
        data["status"] = "completed"
        # Ensure processed_images == total_images at the end
        data["processed_images"] = data["total_images"]
        f.seek(0)
        json.dump(data, f)
        f.truncate()
        upload_embedding_on_firebase(target_folder, status_file)





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
            target_folder = f"uploads/{wedding_folder_id}"
            upload_embedding_on_firebase(folder_path, status_file)

        else:
            data["status"] = "processing"
        # Save the updated status
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    return JSONResponse(data)




import requests
import time
API_KEY = "rpa_L8PXG3HWRZCFWJKQCCU1IFJ8W9PNBU60MZC9XUMN2u1ydp"
POD_ID = "zqgxtfm71sydwm"  # Get this from RunPod dashboard

BASE_URL  = "https://rest.runpod.io/v1/pods"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}



def get_pod_status(pod_id):
    url = f"{BASE_URL}/{pod_id}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("desiredStatus")
        print(f"Pod current status: {status}")
        return status
    else:
        print(f"Failed to get pod status: {resp.status_code} {resp.text}")
        return None

@app.get("/check_pod_status")
async def check_status():
    POD_ID = "zqgxtfm71sydwm"
    pod_status  =  get_pod_status(POD_ID)
    dict = {
        "pod_status":pod_status
    }

    return JSONResponse(dict)

def stop_pod(pod_id):
    url  = f"{BASE_URL}/{pod_id}/stop"
    resp = requests.post(url, headers=HEADERS)
    print(f"Stop pod response: {resp.status_code} {resp.text}")
    return resp.status_code == 200

@app.get("/stop_pod")
async def check_status():
    POD_ID = "zqgxtfm71sydwm"
    pod_status  =  stop_pod(POD_ID)
    dict = {
        "stop_status":pod_status
    }

    return JSONResponse(dict)

def start_pod(pod_id):

    url  = f"{BASE_URL}/{pod_id}/start"
    resp = requests.post(url, headers=HEADERS)
    print(f"Start pod response: {resp.status_code} {resp.text}")
    return resp.status_code == 200



@app.get("/start_pod")
async def check_status():
    POD_ID = "zqgxtfm71sydwm"
    info  = get_pod_status(POD_ID)
    if info ==  "RUNNING":
        dict = {
            "start_status": True
        }
    else:

        pod_status  =  start_pod(POD_ID)
        dict = {
            "start_status":pod_status
        }

    return JSONResponse(dict)




from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import urllib.parse
@app.post("/find_person")
async def find_face_fast(
    input_img: UploadFile,
    wedding_folder_id: str = Form(...)
):
    # === 1. Save input image temporarily ===
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_input:
        shutil.copyfileobj(input_img.file, tmp_input)
        img_path = tmp_input.name

    try:
        # === 2. Extract target face embedding ===
        target_img_np = cv2.imread(img_path)
        if target_img_np is None:
            return JSONResponse({"message": "Invalid image", "match_list": []})

        faces = app_faces.get(target_img_np)
        if not faces:
            return JSONResponse({"message": "No face detected", "match_list": []})

        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        target_embedding = largest_face.embedding.reshape(1, -1)

        # === 3. Define Firebase paths ===
        embeddings_folder = f"uploads/{wedding_folder_id}_embeddings"
        status_json_path = f"uploads/{wedding_folder_id}/status.json"

        # # === 4. Download status.json to get image URL mapping ===
        # try:
        #     status_blob = bucket.blob(status_json_path)
        #     status_data = status_blob.download_as_string()
        #     import json
        #     status = json.loads(status_data)
        #     all_images = status.get("all_images", [])
        #     if not all_images:
        #         return JSONResponse({"message": "No images in status.json", "match_list": []})
        # except Exception as e:
        #     return JSONResponse({"message": f"Failed to load status.json: {str(e)}", "match_list": []})

        # Map base filename (e.g., DSC00806) -> public URL
        import json
        with open(status_json_path, "r") as f:
            json_data = json.load(f)
        all_images = json_data.get("all_images", [])
        name_to_url = {}
        for url in all_images:
            # Extract filename from URL
            filename = url.split("o/")[1].split("?")[0]
            filename = requests.utils.unquote(filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            name_to_url[base_name] = url

        # === 5. List and download all .pkl embedding files from Firebase ===
        # blobs = bucket.list_blobs(prefix=embeddings_folder + "/")
        # pkl_blobs = [b for b in blobs if b.name.endswith('.pkl')]
        #
        # if not pkl_blobs:
        #     return JSONResponse({"message": "No embedding files found", "match_list": []})

        # Download all embeddings into memory (or temp files if too big)
        embs_list = os.listdir(embeddings_folder)
        if not embs_list:
            return JSONResponse({"message": "No embedding files found", "match_list": []})
        loaded_data = []
        for blob in embs_list:
            try:
                emb_path = os.path.join(embeddings_folder, blob)
                # pkl_data = blob.download_as_bytes()
                with open(emb_path, "rb") as f:
                    embeddings = pickle.load(f)
                # embeddings = pickle.loads(pkl_data)
                if isinstance(embeddings, list):
                    embeddings = np.array(embeddings)
                else:
                    embeddings = np.array([embeddings])





                # Decode URL-encoded pickle path and get base name
                blob_path_decoded = urllib.parse.unquote(blob)  # ai_photos/studio_Aman Studio/.../DSC00806_1.pkl
                blob_file_name = os.path.basename(blob_path_decoded)
                blob_base_name = os.path.splitext(blob_file_name)[0]  # DSC00806_1
                # Remove numeric suffix if exists
                if '_' in blob_base_name:
                    blob_base_name = blob_base_name.split('_')[0]  # DSC00806

                # Find matching image
                matched_image = None
                for img_url in all_images:
                    # Remove query params and decode
                    img_path = urllib.parse.unquote(img_url.split('?')[0])
                    img_file_name = os.path.basename(img_path)
                    img_base_name = os.path.splitext(img_file_name)[0]  # DSC00806
                    if img_base_name == blob_base_name:
                        matched_image = img_url
                        break

                if matched_image:

                    loaded_data.append((matched_image, embeddings))


            except Exception as e:
                print(f"Failed to load: {e}")

        # === 6. Parallel similarity search ===
        batch_size = 256
        matches = []

        def process_batch(batch_data):
            local_matches = []
            for base_name, emb_array in batch_data:
                print("basename:",base_name)
                sims = cosine_similarity(target_embedding, emb_array)[0]
                best_sim = np.max(sims)
                if best_sim >= SIMILARITY_THRESHOLD:
                    # if base_name in name_to_url:
                    local_matches.append(base_name)
            return local_matches

        num_workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(loaded_data), batch_size):
                batch = loaded_data[i:i + batch_size]
                print("batch:",batch)
                futures.append(executor.submit(process_batch, batch))

            for future in as_completed(futures):
                matches.extend(future.result())

        # === 7. Return sorted matches ===
        # matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)[:50]  # top 50

        return JSONResponse({
            "status": True,
            "message": "Matches found" if matches else "No matches found",
            "match_count": len(matches),
            "matches": matches
        })

    finally:
        # Clean up temp input file
        if os.path.exists(img_path):
            os.unlink(img_path)





if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=False)

