
import os
import json
import sqlite3
import tempfile
import webbrowser
from threading import Timer
from functools import wraps
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import tensorflow as tf
from pydub import AudioSegment
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "learning-platform-dev-secret-change-later"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "users.db")

MAX_AUDIO_SIZE = 25 * 1024 * 1024
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_VIDEO_SIZE = 25 * 1024 * 1024

app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_SIZE


# ============================================================
# ISL SIGN-TO-TEXT MODEL
# ============================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")
ISL_MODEL_PATH = os.path.join(MODEL_DIR, "isl_model")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, "class_mapping.json")

_isl_model = None
_class_mapping = None


def load_isl_model():
    """Load the model and label mapping when first needed."""
    global _isl_model, _class_mapping

    saved_model_file = os.path.join(
        ISL_MODEL_PATH, "saved_model.pb"
    )

    if not os.path.isfile(saved_model_file):
        raise FileNotFoundError(
            f"ISL SavedModel not found: {ISL_MODEL_PATH}"
        )

    if not os.path.isfile(CLASS_MAPPING_PATH):
        raise FileNotFoundError(
            f"ISL class mapping not found: {CLASS_MAPPING_PATH}"
        )

    if _isl_model is None:
        _isl_model = tf.saved_model.load(ISL_MODEL_PATH)

    if _class_mapping is None:
        with open(
            CLASS_MAPPING_PATH, "r", encoding="utf-8"
        ) as file:
            _class_mapping = json.load(file)

    return _isl_model, _class_mapping


def recognize_isl_image(image_bytes):
    """Recognize a static alphabet sign from an uploaded image."""

    if not os.path.isfile(HAND_MODEL_PATH):
        raise FileNotFoundError(
            f"MediaPipe hand model not found: {HAND_MODEL_PATH}"
        )

    encoded_image = np.frombuffer(
        image_bytes, dtype=np.uint8
    )
    image = cv2.imdecode(
        encoded_image, cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "The image could not be read. Please try another image."
        )

    rgb_image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_image)
    )

    with vision.HandLandmarker.create_from_options(
        options
    ) as detector:
        results = detector.detect(mp_image)

    detected_hands = results.hand_landmarks

    if not detected_hands:
        raise ValueError(
            "No hand was detected. Please try a clearer image "
            "with your hand fully visible."
        )

    # Keep detected hands in left-to-right order.
    detected_hands = sorted(
        detected_hands,
        key=lambda hand: float(
            np.mean([point.x for point in hand])
        )
    )

    landmarks = []

    for hand in detected_hands[:2]:
        for point in hand:
            landmarks.extend([
                point.x, point.y, point.z
            ])

    # The model expects 126 values:
    # 42 landmarks × 3 coordinates.
    if len(landmarks) == 63:
        landmarks.extend([0.0] * 63)

    if len(landmarks) != 126:
        raise ValueError(
            "Could not prepare the hand data. Please try another image."
        )

    landmarks = np.asarray(
        landmarks, dtype=np.float32
    ).reshape(42, 3)

    # Normalize landmarks relative to the first wrist.
    landmarks -= landmarks[0].copy()

    palm_size = float(
        np.linalg.norm(landmarks[9])
    )

    if palm_size > 0:
        landmarks /= palm_size

    input_values = landmarks.flatten().reshape(1, 126)

    model, mapping = load_isl_model()
    infer = model.signatures["serving_default"]

    outputs = infer(
        input_1=tf.convert_to_tensor(
            input_values,
            dtype=tf.float32
        )
    )

    scores = next(iter(outputs.values())).numpy()[0]
    class_index = int(np.argmax(scores))

    letter = mapping.get(str(class_index), "Unknown")
    confidence = float(scores[class_index]) * 100.0

    return letter, confidence


def recognize_isl_video(video_bytes, extension=".mp4"):
    """
    Recognize alphabet signs from sampled video frames.

    Samples video frames, smooths short prediction changes, and
    collapses consecutive repeated labels into a sign sequence.

    This uses the existing static-image model. It does not perform
    full word or sentence translation.
    """

    if not video_bytes:
        raise ValueError("The uploaded video is empty.")

    allowed_extensions = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm"
    }

    extension = extension.lower()

    if not extension.startswith("."):
        extension = "." + extension

    if extension not in allowed_extensions:
        raise ValueError(
            "Unsupported video type. Upload MP4, AVI, MOV, MKV, or WEBM."
        )

    if not os.path.isfile(HAND_MODEL_PATH):
        raise FileNotFoundError(
            f"MediaPipe hand model not found: {HAND_MODEL_PATH}"
        )

    temp_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_video:
            temp_path = temp_video.name
            temp_video.write(video_bytes)

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "Unable to open this video. Try a short MP4 video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        if total_frames <= 0:
            raise ValueError(
                "The uploaded video has no readable frames."
            )

        # Sample approximately five frames per second.
        frame_step = max(1, round(fps / 5))

        # Limit the total number of processed frames.
        max_sampled_frames = 600
        estimated_samples = (
            total_frames + frame_step - 1
        ) // frame_step

        if estimated_samples > max_sampled_frames:
            frame_step = max(
                frame_step,
                (
                    total_frames + max_sampled_frames - 1
                ) // max_sampled_frames
            )

        observations = []
        frame_index = 0
        sampled_frames = 0

        while sampled_frames < max_sampled_frames:
            success, frame = cap.read()

            if not success:
                break

            if frame_index % frame_step == 0:
                sampled_frames += 1

                # Resize large frames to reduce processing time.
                height, width = frame.shape[:2]
                largest_side = max(height, width)

                if largest_side > 960:
                    scale = 960 / largest_side
                    frame = cv2.resize(
                        frame,
                        (
                            int(width * scale),
                            int(height * scale)
                        ),
                        interpolation=cv2.INTER_AREA
                    )

                encoded_success, encoded_frame = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92]
                )

                if not encoded_success:
                    observations.append(None)
                else:
                    try:
                        letter, confidence = recognize_isl_image(
                            encoded_frame.tobytes()
                        )

                        if (
                            letter
                            and str(letter).upper() != "UNKNOWN"
                            and np.isfinite(confidence)
                            and confidence >= 55
                        ):
                            observations.append(
                                (
                                    str(letter).upper(),
                                    float(confidence)
                                )
                            )
                        else:
                            observations.append(None)

                    except ValueError:
                        # Ignore frames where a clear hand is not found.
                        observations.append(None)

            frame_index += 1

        if not observations or not any(
            item is not None for item in observations
        ):
            raise ValueError(
                "No clear signs were recognized. Try a well-lit video "
                "with your hand clearly visible."
            )

        # --------------------------------------------------------
        # SMOOTH PREDICTIONS
        # Use neighbouring valid predictions, but do not smooth
        # across a missing-frame gap.
        # --------------------------------------------------------
        smoothed = []

        for index, current in enumerate(observations):
            if current is None:
                smoothed.append(None)
                continue

            nearby = [current]

            if index > 0 and observations[index - 1] is not None:
                nearby.append(observations[index - 1])

            if (
                index + 1 < len(observations)
                and observations[index + 1] is not None
            ):
                nearby.append(observations[index + 1])

            counts = Counter(
                label for label, _ in nearby
            )

            highest_count = max(counts.values())

            winners = {
                label
                for label, count in counts.items()
                if count == highest_count
            }

            # Prefer the current prediction when tied.
            if current[0] in winners:
                winner = current[0]
            else:
                winner = sorted(winners)[0]

            winning_confidences = [
                confidence
                for label, confidence in nearby
                if label == winner
            ]

            average_confidence = (
                sum(winning_confidences)
                / len(winning_confidences)
            )

            smoothed.append(
                (winner, average_confidence)
            )

        # --------------------------------------------------------
        # GROUP CONSECUTIVE MATCHING PREDICTIONS INTO SEGMENTS
        # --------------------------------------------------------
        segments = []
        current_label = None
        current_confidences = []
        current_length = 0

        def finish_segment():
            if current_label is not None:
                segments.append({
                    "label": current_label,
                    "frames": current_length,
                    "confidence": (
                        sum(current_confidences)
                        / len(current_confidences)
                    )
                })

        for item in smoothed:
            if item is None:
                finish_segment()
                current_label = None
                current_confidences = []
                current_length = 0
                continue

            label, confidence = item

            if label == current_label:
                current_length += 1
                current_confidences.append(confidence)
            else:
                finish_segment()
                current_label = label
                current_confidences = [confidence]
                current_length = 1

        finish_segment()

        # Filter out isolated predictions that appear in only
        # one sampled frame. Adjust this threshold if needed.
        minimum_segment_frames = 2

        stable_segments = [
            segment
            for segment in segments
            if segment["frames"] >= minimum_segment_frames
        ]

        if not stable_segments:
            raise ValueError(
                "No stable signs were recognized. Try a clearer or "
                "longer video with your hand fully visible."
            )

        return " ".join(
            segment["label"]
            for segment in stable_segments
        )

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                app.logger.warning(
                    "Could not remove temporary video file: %s",
                    temp_path
                )


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get(
            "email", ""
        ).strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            return """
                <h2>Login Error</h2>
                <p>Please enter your email and password.</p>
                <a href="/login">Back to Login</a>
            """, 400

        conn = get_db_connection()
        try:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()
        finally:
            conn.close()

        if user and check_password_hash(
            user["password_hash"], password
        ):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]
            return redirect(url_for("home"))

        return """
            <h2>Login Failed</h2>
            <p>Invalid email or password.</p>
            <a href="/login">Try Again</a>
        """, 401

    return render_template("login.html")


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get(
            "email", ""
        ).strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if (
            not name
            or not email
            or not password
            or not confirm_password
        ):
            return """
                <h2>Registration Error</h2>
                <p>Please fill in all fields.</p>
                <a href="/register">Back to Register</a>
            """, 400

        if password != confirm_password:
            return """
                <h2>Registration Error</h2>
                <p>Passwords do not match.</p>
                <a href="/register">Try Again</a>
            """, 400

        password_hash = generate_password_hash(password)
        conn = get_db_connection()

        try:
            conn.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                (name, email, password_hash)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return """
                <h2>Registration Error</h2>
                <p>An account with this email already exists.</p>
                <a href="/login">Go to Login</a>
            """, 409
        finally:
            conn.close()

        return redirect(url_for("login"))

    return render_template("register.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
@login_required
def profile():
    return render_template(
        "profile.html",
        name=session.get("user_name"),
        email=session.get("user_email")
    )


# ============================================================
# LEARNING METHODS
# ============================================================

@app.route("/learn")
@login_required
def learn():
    return render_template("learn.html")


@app.route("/learning-methods")
@login_required
def learning_methods():
    return redirect(url_for("learn"))


# ============================================================
# TEXT LESSONS
# ============================================================

@app.route("/text-lessons")
@login_required
def text_lessons():
    return render_template("lessons/text_lessons.html")


@app.route("/ai-lesson")
@login_required
def ai_lesson():
    return render_template("lessons/ai_lesson.html")


@app.route("/ml-lesson")
@login_required
def ml_lesson():
    return render_template("lessons/ml_lesson.html")


@app.route("/computer-lesson")
@login_required
def computer_lesson():
    return render_template("lessons/computer_lesson.html")


@app.route("/web-lesson")
@login_required
def web_lesson():
    return render_template("lessons/web_lesson.html")


# ============================================================
# VIDEO LEARNING
# ============================================================

@app.route("/video-learning")
@login_required
def video_learning():
    return render_template(
        "video_learning/video_learning.html"
    )


@app.route("/ai-video")
@login_required
def ai_video():
    return render_template("video_learning/ai_video.html")


@app.route("/ml-video")
@login_required
def ml_video():
    return render_template("video_learning/ml_video.html")


@app.route("/computer-fundamentals")
@login_required
def computer_fundamentals():
    return render_template(
        "video_learning/computer_fundamentals.html"
    )


@app.route("/web-development-video")
@login_required
def web_development_video():
    return render_template(
        "video_learning/web_development.html"
    )


# ============================================================
# QUIZZES
# ============================================================

def calculate_quiz_score(answers):
    return sum(
        1 for question, correct_answer in answers.items()
        if request.form.get(question) == correct_answer
    )


@app.route("/ai-quiz", methods=["GET", "POST"])
@login_required
def ai_quiz():
    if request.method == "POST":
        answers = {
            "q1": "b",
            "q2": "a",
            "q3": "a",
            "q4": "d",
            "q5": "a"
        }
        return render_template(
            "quizzes/ai_quiz_result.html",
            score=calculate_quiz_score(answers),
            total=len(answers)
        )
    return render_template("quizzes/ai_quiz.html")


@app.route("/ml-quiz", methods=["GET", "POST"])
@login_required
def ml_quiz():
    if request.method == "POST":
        answers = {
            "q1": "a", "q2": "a", "q3": "a",
            "q4": "a", "q5": "a"
        }
        return render_template(
            "quizzes/ml_quiz_result.html",
            score=calculate_quiz_score(answers),
            total=len(answers)
        )
    return render_template("quizzes/ml_quiz.html")


@app.route("/web-quiz", methods=["GET", "POST"])
@login_required
def web_quiz():
    if request.method == "POST":
        answers = {
            "q1": "a", "q2": "a", "q3": "a",
            "q4": "a", "q5": "a"
        }
        return render_template(
            "quizzes/web_quiz_result.html",
            score=calculate_quiz_score(answers),
            total=len(answers)
        )
    return render_template("quizzes/web_quiz.html")


@app.route("/computer-quiz", methods=["GET", "POST"])
@login_required
def computer_quiz():
    if request.method == "POST":
        answers = {
            "q1": "a", "q2": "a", "q3": "a",
            "q4": "a", "q5": "a"
        }
        return render_template(
            "quizzes/computer_quiz_result.html",
            score=calculate_quiz_score(answers),
            total=len(answers)
        )
    return render_template("quizzes/computer_quiz.html")


# ============================================================
# VISUAL LEARNING
# ============================================================

@app.route("/visual-learning")
@login_required
def visual_learning():
    return render_template("visual_learning.html")


@app.route("/visual-ai-lesson")
@login_required
def visual_ai_lesson():
    return render_template("lessons/visual_ai_lesson.html")


@app.route("/visual-ml-lesson")
@login_required
def visual_ml_lesson():
    return render_template("lessons/visual_ml_lesson.html")


@app.route("/visual-computer-lesson")
@login_required
def visual_computer_lesson():
    return render_template("lessons/visual_computer_lesson.html")


@app.route("/visual-web-lesson")
@login_required
def visual_web_lesson():
    return render_template("lessons/visual_web_lesson.html")


# ============================================================
# SIGN LANGUAGE
# ============================================================

@app.route("/sign-language")
@login_required
def sign_language():
    return render_template(
        "sign_language/sign_language.html"
    )


@app.route("/sign-language/quiz")
@login_required
def sign_language_quiz():
    return render_template(
        "sign_language/sign_quiz.html"
    )


# ============================================================
# AUDIO TO TEXT
# ============================================================

@app.route("/audio-to-text")
@login_required
def audio_to_text():
    return render_template(
        "conversions/audio_to_text.html"
    )


def transcribe_uploaded_audio(uploaded_file, language):
    extension = os.path.splitext(
        uploaded_file.filename
    )[1].lower()

    allowed_extensions = {
        ".wav", ".mp3", ".ogg", ".flac", ".m4a"
    }

    if extension not in allowed_extensions:
        raise ValueError(
            "Unsupported file type. Upload WAV, MP3, OGG, FLAC, or M4A."
        )

    temp_input = None
    temp_wav = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:
            uploaded_file.save(temp.name)
            temp_input = temp.name

        audio = AudioSegment.from_file(temp_input)

        if len(audio) == 0:
            raise ValueError(
                "The uploaded audio file is empty."
            )

        audio = audio.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp:
            temp_wav = temp.name

        audio.export(temp_wav, format="wav")

        recognizer = sr.Recognizer()

        with sr.AudioFile(temp_wav) as source:
            recording = recognizer.record(source)

        return recognizer.recognize_google(
            recording,
            language=language
        ).strip()

    finally:
        for path in (temp_input, temp_wav):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    app.logger.warning(
                        "Could not remove temporary audio file: %s",
                        path
                    )


@app.route("/convert-audio-file", methods=["POST"])
@login_required
def convert_audio_file():
    uploaded_file = request.files.get("audio_file")
    raw_language = request.form.get(
        "language", "en-IN"
    ).strip()

    language_aliases = {
        "en-IN": "en-IN",
        "en-US": "en-US",
        "en_US": "en-US",
        "en": "en-IN",
        "english": "en-IN",
        "english (india)": "en-IN",
        "english (us)": "en-US",
        "ta-IN": "ta-IN",
        "ta": "ta-IN",
        "tamil": "ta-IN",
        "hi-IN": "hi-IN",
        "hi": "hi-IN",
        "hindi": "hi-IN"
    }

    language = language_aliases.get(
        raw_language,
        language_aliases.get(raw_language.lower())
    )

    if language is None:
        return jsonify({
            "error": "Unsupported language. Choose English, Tamil, or Hindi."
        }), 400

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({
            "error": "Please choose an audio file."
        }), 400

    try:
        text = transcribe_uploaded_audio(
            uploaded_file, language
        )

        if not text:
            return jsonify({
                "error": "No speech was recognized."
            }), 422

        return jsonify({
            "text": text,
            "language": language
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except sr.UnknownValueError:
        return jsonify({
            "error": "Speech could not be recognized clearly."
        }), 422

    except sr.RequestError:
        app.logger.exception(
            "Speech recognition service failed"
        )
        return jsonify({
            "error": (
                "Speech recognition service unavailable. "
                "Check your internet connection."
            )
        }), 502

    except Exception:
        app.logger.exception(
            "Audio-to-text conversion failed"
        )
        return jsonify({
            "error": (
                "Could not process the audio file. "
                "Check the file and FFmpeg installation."
            )
        }), 400


# ============================================================
# AUDIO TO SIGN LANGUAGE
# ============================================================

@app.route("/audio-to-sign")
@login_required
def audio_to_sign():
    return render_template(
        "conversions/audio_to_sign.html"
    )


@app.route("/audio-to-sign/convert", methods=["POST"])
@login_required
def audio_to_sign_convert():
    uploaded_file = request.files.get("audio_file")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({
            "success": False,
            "message": "Please select an audio file."
        }), 400

    try:
        recognized_text = transcribe_uploaded_audio(
            uploaded_file, "en-IN"
        )

        if not recognized_text:
            return jsonify({
                "success": False,
                "message": (
                    "No speech was recognized. "
                    "Try a clearer recording."
                )
            }), 422

        return jsonify({
            "success": True,
            "text": recognized_text,
            "message": "Speech recognized successfully."
        })

    except ValueError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

    except sr.UnknownValueError:
        return jsonify({
            "success": False,
            "message": "Speech could not be recognized clearly."
        }), 422

    except sr.RequestError:
        app.logger.exception(
            "Audio-to-sign speech recognition failed"
        )
        return jsonify({
            "success": False,
            "message": (
                "Speech recognition is unavailable. "
                "Check your internet connection."
            )
        }), 502

    except Exception:
        app.logger.exception(
            "Audio-to-sign conversion failed"
        )
        return jsonify({
            "success": False,
            "message": (
                "Could not process the audio file. "
                "Check FFmpeg and try again."
            )
        }), 400


# ============================================================
# TEXT TO SIGN
# ============================================================

@app.route("/text-to-sign")
@login_required
def text_to_sign():
    return render_template(
        "conversions/text_to_sign.html"
    )


@app.route("/text-to-sign/convert", methods=["POST"])
@login_required
def convert_text_to_sign():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str):
        return jsonify({
            "error": "Text must be a string."
        }), 400

    text = text.strip()

    if not text:
        return jsonify({
            "error": "Please enter some text."
        }), 400

    return jsonify({
        "message": "Text received successfully.",
        "text": text
    })


# ============================================================
# SIGN TO TEXT — IMAGE AND VIDEO
# ============================================================

@app.route("/sign-to-text", methods=["GET", "POST"])
@login_required
def sign_to_text():
    recognized_text = None

    if request.method == "POST":
        image_file = request.files.get("sign_image")
        video_file = request.files.get("sign_video")

        # ----------------------------------------------------
        # IMAGE UPLOAD — EXISTING FEATURE
        # ----------------------------------------------------
        if image_file and image_file.filename:
            extension = os.path.splitext(
                image_file.filename
            )[1].lower()

            allowed_extensions = {
                ".png", ".jpg", ".jpeg", ".webp"
            }

            if extension not in allowed_extensions:
                recognized_text = (
                    "Unsupported image type. Upload PNG, JPG, JPEG, "
                    "or WEBP."
                )
            else:
                try:
                    image_bytes = image_file.read()

                    if not image_bytes:
                        raise ValueError(
                            "The uploaded image is empty."
                        )

                    if len(image_bytes) > MAX_IMAGE_SIZE:
                        raise ValueError(
                            "Image too large. Upload an image no larger "
                            "than 10 MB."
                        )

                    letter, confidence = recognize_isl_image(
                        image_bytes
                    )

                    recognized_text = (
                        f"Predicted letter: {letter} "
                        f"(model confidence: {confidence:.1f}%)"
                    )

                except ValueError as error:
                    recognized_text = str(error)

                except FileNotFoundError:
                    app.logger.exception(
                        "ISL model file missing"
                    )
                    recognized_text = (
                        "The ISL model files are missing. Check your "
                        "models folder and application terminal."
                    )

                except Exception:
                    app.logger.exception(
                        "ISL image recognition failed"
                    )
                    recognized_text = (
                        "Image recognition failed. Check the model files, "
                        "installed packages, and application terminal."
                    )

        # ----------------------------------------------------
        # VIDEO UPLOAD
        # ----------------------------------------------------
        elif video_file and video_file.filename:
            extension = os.path.splitext(
                video_file.filename
            )[1].lower()

            allowed_video_extensions = {
                ".mp4", ".avi", ".mov", ".mkv", ".webm"
            }

            if extension not in allowed_video_extensions:
                recognized_text = (
                    "Unsupported video type. Upload MP4, AVI, MOV, "
                    "MKV, or WEBM."
                )
            else:
                try:
                    video_bytes = video_file.read()

                    if not video_bytes:
                        raise ValueError(
                            "The uploaded video is empty."
                        )

                    if len(video_bytes) > MAX_VIDEO_SIZE:
                        raise ValueError(
                            "Video too large. Upload a video no larger "
                            "than 25 MB."
                        )

                    recognized_sequence = recognize_isl_video(
                        video_bytes, extension
                    )

                    recognized_text = (
                        "Recognized sign sequence: "
                        + recognized_sequence
                    )

                except ValueError as error:
                    recognized_text = str(error)

                except FileNotFoundError:
                    app.logger.exception(
                        "ISL model file missing"
                    )
                    recognized_text = (
                        "The ISL model files are missing. Check your "
                        "models folder and application terminal."
                    )

                except Exception:
                    app.logger.exception(
                        "ISL video recognition failed"
                    )
                    recognized_text = (
                        "Video recognition failed. Check the video format, "
                        "model files, and application terminal."
                    )

        else:
            recognized_text = (
                "Please upload an image or a video."
            )

    return render_template(
        "conversions/sign_to_text.html",
        recognized_text=recognized_text
    )


# ============================================================
# MAXIMUM UPLOAD SIZE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    if request.path == "/audio-to-sign/convert":
        return jsonify({
            "success": False,
            "message": (
                "Please choose an audio file no larger than 25 MB."
            )
        }), 413

    if request.path == "/convert-audio-file":
        return jsonify({
            "error": (
                "Please choose an audio file no larger than 25 MB."
            )
        }), 413

    if request.path == "/sign-to-text":
        return (
            "File too large. Images must be no larger than 10 MB "
            "and videos no larger than 25 MB.",
            413
        )

    return (
        "File too large. Maximum upload size is 25 MB.",
        413
    )


# ============================================================
# OPEN BROWSER AND START APP
# ============================================================


def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:5000")


if __name__ == "__main__":
    init_db()
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
