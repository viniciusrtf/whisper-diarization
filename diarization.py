import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from collections import defaultdict

# 1) Load Whisper ASR model
whisper_model = whisper.load_model("small", device="cuda")

# 2) Transcribe audio with timestamped segments
audio_file = "original.wav"
whisper_result = whisper_model.transcribe(
    audio_file,
    word_timestamps=True
)
segments = whisper_result["segments"]

# 3) Load pyannote diarization pipeline v3.1
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = pipeline.to(device)

# 4) Run diarization with progress hook and 2 speakers constraint
with ProgressHook() as hook:
    diarization = pipeline(
        audio_file,
        hook=hook,
        num_speakers=2  # constrain to exactly 2 speakers ([atyun.com](https://www.atyun.com/models/info/philschmid/pyannote-speaker-diarization-endpoint.html?lang=en&utm_source=chatgpt.com))
    )

# 5) Align segments to speakers
def who_speaks(t):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= t < turn.end:
            return speaker
    return "Unknown"

speaker_transcripts = defaultdict(list)
for seg in segments:
    midpoint = (seg["start"] + seg["end"]) / 2
    spk = who_speaks(midpoint)
    speaker_transcripts[spk].append(seg)

# 6) Print transcripts by speaker
for spk, segs in speaker_transcripts.items():
    print(f"\n>>> {spk}")
    for s in segs:
        print(f"[{s['start']:.1f}s–{s['end']:.1f}s] {s['text']}")

