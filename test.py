from dotenv import load_dotenv
load_dotenv()
from utils.audio_processor import process_input
from core.transcriber import transcribe_all

load_dotenv()

source =  ""

chunks = process_input(source)

print(transcribe_all(chunks,hinglish=True))