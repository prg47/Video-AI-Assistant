import whisper
import os
import json
from datetime import datetime

from sarvamai import SarvamAI

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

OUTPUT_DIR = "sarvam_output"

_model = None


# Initialize Sarvam client
sarvam_client = None

if SARVAM_API_KEY:

    sarvam_client = SarvamAI(
        api_subscription_key=SARVAM_API_KEY
    )


def load_model():
    """
    Loads Whisper model lazily
    """

    global _model

    if _model is None:

        print("Loading Whisper model...")

        _model = whisper.load_model(
            WHISPER_MODEL
        )

        print("Whisper model loaded successfully")

    return _model


def transcribe_with_whisper(
    chunk_path: str,
    translate: bool = False
) -> str:
    """
    Uses Whisper for English transcription
    """

    model = load_model()

    task = (
        "translate"
        if translate
        else "transcribe"
    )

    result = model.transcribe(
        chunk_path,
        task=task
    )

    return result["text"]


def transcribe_with_sarvam_batch(
    audio_paths: list[str]
) -> str:
    """
    Uses Sarvam Batch API
    for Hindi/Hinglish audio
    """

    if not sarvam_client:

        raise ValueError(
            "SARVAM_API_KEY not found"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        f"[{datetime.now()}] "
        "Creating batch job..."
    )

    job = (
        sarvam_client
        .speech_to_text_job
        .create_job(
            model="saaras:v3",
            mode="transcribe",
            language_code="unknown",
            with_diarization=False
        )
    )

    print(
        f"[{datetime.now()}] "
        "Uploading files..."
    )

    job.upload_files(
        file_paths=audio_paths
    )

    print(
        f"[{datetime.now()}] "
        "Starting job..."
    )

    job.start()

    print(
        f"[{datetime.now()}] "
        "Waiting for completion..."
    )

    # SDK handles polling internally
    job.wait_until_complete()

    print(
        f"[{datetime.now()}] "
        "Job completed!"
    )

    print(
        f"[{datetime.now()}] "
        "Fetching results..."
    )

    file_results = job.get_file_results()

    print("\nSuccessful Files:")
    print(
        file_results.get(
            "successful",
            []
        )
    )

    print("\nFailed Files:")
    print(
        file_results.get(
            "failed",
            []
        )
    )

    print(
        f"[{datetime.now()}] "
        "Downloading outputs..."
    )

    job.download_outputs(
        output_dir=OUTPUT_DIR
    )

    full_text = ""

    # Read downloaded transcripts
    for file_name in os.listdir(
        OUTPUT_DIR
    ):

        if not file_name.endswith(
            ".json"
        ):
            continue

        file_path = os.path.join(
            OUTPUT_DIR,
            file_name
        )

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            transcript = (
                data.get("transcript")
                or data.get("text")
                or ""
            )

            full_text += (
                transcript + " "
            )

        except Exception as e:

            print(
                f"Failed reading "
                f"{file_name}: {e}"
            )

    return full_text.strip()


def transcribe_chunk(
    chunk_path: str,
    translate: bool = False,
    hinglish: bool = False
) -> str:
    """
    Main transcription function
    """

    if hinglish:

        return (
            transcribe_with_sarvam_batch(
                [chunk_path]
            )
        )

    return transcribe_with_whisper(
        chunk_path,
        translate=translate
    )


def transcribe_all(
    chunks: list,
    translate: bool = False,
    hinglish: bool = False
) -> str:
    """
    Main transcription pipeline

    hinglish=True:
        Uses Sarvam Batch API

    hinglish=False:
        Uses Whisper
    """

    # Sarvam batch mode
    if hinglish:

        print(
            "Using Sarvam Batch API..."
        )

        return (
            transcribe_with_sarvam_batch(
                chunks
            )
        )

    # Whisper mode
    full_transcript = ""

    for i, chunk in enumerate(
        chunks
    ):

        print(
            f"Transcribing chunk "
            f"{i + 1}"
        )

        text = transcribe_chunk(
            chunk,
            translate=translate
        )

        full_transcript += (
            text + " "
        )

    print(
        "Transcription completed"
    )

    return full_transcript.strip()




