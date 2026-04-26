import os
import base64
import time
import sqlite3
import numpy as np
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import cv2
import face_recognition
from PIL import Image
from dotenv import load_dotenv
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

load_dotenv()
limiter = Limiter(key_func=get_remote_address)

SECRET_KEY = "faceid-super-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")



mail_config = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)

ALERT_TO = os.getenv("ALERT_TO")

app = FastAPI(title="FaceID API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path("facedb.sqlite")
FACES_DIR = Path("registered_faces")
FACES_DIR.mkdir(exist_ok=True)
UNKNOWN_DIR = Path("unknown_faces")
UNKNOWN_DIR.mkdir(exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            label TEXT,
            photo_path TEXT,
            encoding BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT,
            confidence REAL,
            source TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (person_id) REFERENCES persons(id)
        );
        CREATE TABLE IF NOT EXISTS person_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            photo_path TEXT,
            encoding BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (person_id) REFERENCES persons(id)
        );
        CREATE TABLE IF NOT EXISTS unknown_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_path TEXT,
            source TEXT,
            timestamp TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


init_db()


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def image_from_upload(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not read image. Please upload a valid JPG or PNG file.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def image_to_base64(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def load_all_encodings():
    conn = get_db()
    rows = conn.execute("""
        SELECT pe.id, pe.person_id, pe.encoding, p.name, p.label
        FROM person_encodings pe
        JOIN persons p ON pe.person_id = p.id
    """).fetchall()
    conn.close()
    known_encodings = []
    known_ids = []
    known_names = []
    known_labels = []
    for row in rows:
        enc = np.frombuffer(row["encoding"], dtype=np.float64)
        known_encodings.append(enc)
        known_ids.append(row["person_id"])
        known_names.append(row["name"])
        known_labels.append(row["label"] or "")
    return known_encodings, known_ids, known_names, known_labels


async def send_unknown_alert(confidence_scores: list, count: int):
    try:
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto;">
            <div style="background: #6366f1; padding: 20px; border-radius: 10px 10px 0 0;">
                <h2 style="color: white; margin: 0;">FaceID Alert</h2>
            </div>
            <div style="background: #f9fafb; padding: 24px; border-radius: 0 0 10px 10px; border: 1px solid #e5e7eb;">
                <h3 style="color: #111827;">Unknown face detected</h3>
                <p style="color: #6b7280;">An unrecognized person was detected by your FaceID system.</p>
                <div style="background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 16px 0;">
                    <p style="margin: 0 0 8px;"><strong>Faces detected:</strong> {count}</p>
                    <p style="margin: 0;"><strong>Time:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                </div>
                <p style="color: #6b7280; font-size: 13px;">This alert was sent automatically by your FaceID system.</p>
            </div>
        </div>
        """
        message = MessageSchema(
            subject="FaceID Alert — Unknown face detected",
            recipients=[ALERT_TO],
            body=html,
            subtype="html",
        )
        fm = FastMail(mail_config)
        await fm.send_message(message)
    except Exception as e:
        print(f"Email alert failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "message": "FaceID API is running!"}


@app.post("/login")
@limiter.limit("10/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username=?", (form_data.username,)
    ).fetchone()
    conn.close()
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}


@app.get("/me")
def get_me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}


@app.post("/register")
@limiter.limit("20/minute")
async def register_face(request: Request,
    name: str = Form(...),
    label: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    file_bytes = await file.read()
    img_rgb = image_from_upload(file_bytes)

    face_locations = face_recognition.face_locations(img_rgb, model="hog")
    if not face_locations:
        raise HTTPException(status_code=400, detail="No face detected. Use a clear frontal photo.")
    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Upload a photo with only one face.")

    encodings = face_recognition.face_encodings(img_rgb, face_locations)
    encoding = encodings[0]

    photo_path = str(FACES_DIR / f"{name.replace(' ', '_')}_{int(time.time())}.jpg")
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(photo_path)

    conn = get_db()
    existing = conn.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()

    if existing:
        person_id = existing["id"]
        conn.execute(
            "INSERT INTO person_encodings (person_id, photo_path, encoding) VALUES (?, ?, ?)",
            (person_id, photo_path, encoding.tobytes()),
        )
        photo_count = conn.execute(
            "SELECT COUNT(*) FROM person_encodings WHERE person_id = ?", (person_id,)
        ).fetchone()[0]
        conn.commit()
        conn.close()
        return {
            "success": True,
            "person_id": person_id,
            "name": name,
            "message": f"New photo added for '{name}'. Total photos: {photo_count}",
            "is_new": False,
        }
    else:
        cursor = conn.execute(
            "INSERT INTO persons (name, label, photo_path, encoding) VALUES (?, ?, ?, ?)",
            (name, label, photo_path, encoding.tobytes()),
        )
        person_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO person_encodings (person_id, photo_path, encoding) VALUES (?, ?, ?)",
            (person_id, photo_path, encoding.tobytes()),
        )
        conn.commit()
        conn.close()
        return {
            "success": True,
            "person_id": person_id,
            "name": name,
            "message": f"'{name}' registered successfully.",
            "is_new": True,
        }


@app.post("/recognize")
@limiter.limit("30/minute")
async def recognize_face(request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(0.55),
    source: str = Form("upload"),
    current_user: str = Depends(get_current_user),
):
    file_bytes = await file.read()
    img_rgb = image_from_upload(file_bytes)

    conn = get_db()
    settings = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings").fetchall()}
    conn.close()

    if settings.get("access_control_enabled") == "true":
        now = datetime.utcnow()
        current_hour = now.hour
        current_day = now.isoweekday()
        start_hour = int(settings.get("access_start_hour", 0))
        end_hour = int(settings.get("access_end_hour", 23))
        allowed_days = [int(d) for d in settings.get("access_days", "1,2,3,4,5").split(",")]
        if current_day not in allowed_days:
            raise HTTPException(status_code=403, detail=f"Access not allowed today. Allowed days: Mon-Fri only.")
        if not (start_hour <= current_hour < end_hour):
            raise HTTPException(status_code=403, detail=f"Access not allowed at this time. Allowed hours: {start_hour}:00 - {end_hour}:00 UTC.")

    known_encodings, known_ids, known_names, known_labels = load_all_encodings()

    face_locations = face_recognition.face_locations(img_rgb, model="hog")
    if not face_locations:
        return {"faces": [], "annotated_image": image_to_base64(img_rgb), "count": 0}

    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    results = []
    annotated = img_rgb.copy()
    has_unknown = False

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        person_id = None
        confidence = 0.0
        label = ""

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            similarity = max(0.0, 1.0 - best_dist)

            if best_dist <= threshold:
                name = known_names[best_idx]
                person_id = known_ids[best_idx]
                confidence = round(similarity * 100, 1)
                label = known_labels[best_idx]

                conn = get_db()
                conn.execute(
                    "INSERT INTO recognition_logs (person_id, person_name, confidence, source) VALUES (?, ?, ?, ?)",
                    (person_id, name, confidence, source),
                )
                conn.commit()
                conn.close()

        if name == "Unknown":
            has_unknown = True

        color = (34, 197, 94) if name != "Unknown" else (239, 68, 68)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        cv2.rectangle(annotated, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
        display = f"{name} {confidence:.0f}%" if name != "Unknown" else "Unknown"
        cv2.putText(annotated, display, (left + 4, bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        matched_photo = None
        if person_id:
            conn2 = get_db()
            photo_row = conn2.execute(
                "SELECT photo_path FROM person_encodings WHERE person_id=? LIMIT 1",
                (person_id,)
            ).fetchone()
            conn2.close()
            if photo_row and Path(photo_row["photo_path"]).exists():
                with open(photo_row["photo_path"], "rb") as pf:
                    matched_photo = base64.b64encode(pf.read()).decode()

        face_crop = img_rgb[top:bottom, left:right]
        face_crop_b64 = image_to_base64(face_crop) if face_crop.size > 0 else None

        results.append({
            "name": name,
            "person_id": person_id,
            "confidence": confidence,
            "label": label,
            "box": {"top": top, "right": right, "bottom": bottom, "left": left},
            "face_crop": face_crop_b64,
            "matched_photo": matched_photo,
        })
    if has_unknown:
        asyncio.create_task(send_unknown_alert([], len(face_locations)))
        unknown_path = str(UNKNOWN_DIR / f"unknown_{int(time.time())}.jpg")
        pil_img = Image.fromarray(annotated)
        pil_img.save(unknown_path)
        conn = get_db()
        conn.execute(
            "INSERT INTO unknown_faces (photo_path, source) VALUES (?, ?)",
            (unknown_path, source),
        )
        conn.commit()
        conn.close()

    return {
        "faces": results,
        "annotated_image": image_to_base64(annotated),
        "count": len(results),
    }


@app.get("/persons")
def list_persons(current_user: str = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, label, photo_path, created_at FROM persons ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    persons = []
    for row in rows:
        photo_b64 = None
        if row["photo_path"] and Path(row["photo_path"]).exists():
            with open(row["photo_path"], "rb") as f:
                photo_b64 = base64.b64encode(f.read()).decode()
        persons.append({
            "id": row["id"],
            "name": row["name"],
            "label": row["label"],
            "photo": photo_b64,
            "created_at": row["created_at"],
        })
    return {"persons": persons, "count": len(persons)}


@app.delete("/persons/{person_id}")
def delete_person(person_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT photo_path FROM persons WHERE id=?", (person_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Person not found.")
    if row["photo_path"] and Path(row["photo_path"]).exists():
        os.remove(row["photo_path"])
    conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
    conn.execute("DELETE FROM recognition_logs WHERE person_id=?", (person_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Person deleted."}


@app.get("/logs")
def get_logs(limit: int = 50, current_user: str = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM recognition_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return {"logs": [dict(r) for r in rows]}


@app.get("/stats")
def get_stats(current_user: str = Depends(get_current_user)):
    conn = get_db()
    total_persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    total_recognitions = conn.execute("SELECT COUNT(*) FROM recognition_logs").fetchone()[0]
    today = datetime.utcnow().strftime("%Y-%m-%d")
    today_recognitions = conn.execute(
        "SELECT COUNT(*) FROM recognition_logs WHERE timestamp LIKE ?", (f"{today}%",)
    ).fetchone()[0]
    avg_confidence = conn.execute(
        "SELECT AVG(confidence) FROM recognition_logs WHERE person_name != 'Unknown'"
    ).fetchone()[0]
    top_persons = conn.execute(
        """SELECT person_name, COUNT(*) as count FROM recognition_logs
           WHERE person_name != 'Unknown'
           GROUP BY person_name ORDER BY count DESC LIMIT 5"""
    ).fetchall()
    conn.close()
    return {
        "total_persons": total_persons,
        "total_recognitions": total_recognitions,
        "today_recognitions": today_recognitions,
        "avg_confidence": round(avg_confidence or 0, 1),
        "top_persons": [{"name": r["person_name"], "count": r["count"]} for r in top_persons],
    }


@app.get("/persons/{person_id}/photos")
def get_person_photos(person_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, photo_path, created_at FROM person_encodings WHERE person_id = ? ORDER BY created_at DESC",
        (person_id,)
    ).fetchall()
    conn.close()
    photos = []
    for row in rows:
        photo_b64 = None
        if row["photo_path"] and Path(row["photo_path"]).exists():
            with open(row["photo_path"], "rb") as f:
                photo_b64 = base64.b64encode(f.read()).decode()
        photos.append({
            "id": row["id"],
            "photo": photo_b64,
            "created_at": row["created_at"],
        })
    return {"photos": photos, "count": len(photos)}


@app.delete("/encodings/{encoding_id}")
def delete_encoding(encoding_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT photo_path FROM person_encodings WHERE id=?", (encoding_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found.")
    if row["photo_path"] and Path(row["photo_path"]).exists():
        os.remove(row["photo_path"])
    conn.execute("DELETE FROM person_encodings WHERE id=?", (encoding_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Photo deleted."}


@app.get("/persons/{person_id}/history")
def get_person_history(person_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db()
    person = conn.execute("SELECT name, label FROM persons WHERE id=?", (person_id,)).fetchone()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    logs = conn.execute(
        """SELECT confidence, source, timestamp FROM recognition_logs
           WHERE person_id=? ORDER BY timestamp DESC LIMIT 50""",
        (person_id,)
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) FROM recognition_logs WHERE person_id=?", (person_id,)
    ).fetchone()[0]
    avg_conf = conn.execute(
        "SELECT AVG(confidence) FROM recognition_logs WHERE person_id=?", (person_id,)
    ).fetchone()[0]
    last_seen = conn.execute(
        "SELECT timestamp FROM recognition_logs WHERE person_id=? ORDER BY timestamp DESC LIMIT 1",
        (person_id,)
    ).fetchone()
    conn.close()
    return {
        "person_id": person_id,
        "name": person["name"],
        "label": person["label"],
        "total_recognitions": total,
        "avg_confidence": round(avg_conf or 0, 1),
        "last_seen": last_seen["timestamp"] if last_seen else None,
        "logs": [dict(r) for r in logs],
    }


@app.get("/unknown-faces")
def get_unknown_faces(limit: int = 50, current_user: str = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, photo_path, source, timestamp FROM unknown_faces ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    faces = []
    for row in rows:
        photo_b64 = None
        if row["photo_path"] and Path(row["photo_path"]).exists():
            with open(row["photo_path"], "rb") as f:
                photo_b64 = base64.b64encode(f.read()).decode()
        faces.append({
            "id": row["id"],
            "photo": photo_b64,
            "source": row["source"],
            "timestamp": row["timestamp"],
        })
    return {"faces": faces, "count": len(faces)}


@app.delete("/unknown-faces/{face_id}")
def delete_unknown_face(face_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT photo_path FROM unknown_faces WHERE id=?", (face_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found.")
    if row["photo_path"] and Path(row["photo_path"]).exists():
        os.remove(row["photo_path"])
    conn.execute("DELETE FROM unknown_faces WHERE id=?", (face_id,))
    conn.commit()
    conn.close()
    return {"success": True}
@app.get("/users")
def list_users(current_user: str = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return {"users": [dict(r) for r in rows]}


@app.post("/users")
def create_user(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("admin"),
    current_user: str = Depends(get_current_user),
):
    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists.")
    hashed = pwd_context.hash(password)
    conn.execute(
        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
        (username, hashed, role)
    )
    conn.commit()
    conn.close()
    return {"success": True, "message": f"User '{username}' created successfully."}


@app.delete("/users/{user_id}")
def delete_user(user_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db()
    user = conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user["username"] == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete the default admin.")
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "User deleted."}
@app.get("/stats/confidence-chart")
def get_confidence_chart(days: int = 7, current_user: str = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("""
        SELECT DATE(timestamp) as date, AVG(confidence) as avg_conf, COUNT(*) as count
        FROM recognition_logs
        WHERE confidence > 0
        AND timestamp >= DATE('now', ?)
        GROUP BY DATE(timestamp)
        ORDER BY date ASC
    """, (f'-{days} days',)).fetchall()
    conn.close()
    return {
        "labels": [r["date"] for r in rows],
        "confidence": [round(r["avg_conf"], 1) for r in rows],
        "counts": [r["count"] for r in rows],
    }
@app.get("/settings")
def get_settings(current_user: str = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


@app.post("/settings")
def update_settings(
    key: str = Form(...),
    value: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()
    conn.close()
    return {"success": True, "key": key, "value": value}