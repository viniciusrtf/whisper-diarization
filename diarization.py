import sys
import argparse
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize an audio file using openai-whisper and pyannote.audio"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input audio file"
    )
    parser.add_argument(
        "-o", "--output_file",
        help="Path to write the aligned transcript (default: stdout)",
        default=None
    )
    parser.add_argument(
        "--model_size",
        help="Whisper model size (tiny, base, small, medium, large)",
        default="small"
    )
    parser.add_argument(
        "--min_speakers", type=int,
        help="Minimum number of speakers (default: 2)",
        default=2
    )
    parser.add_argument(
        "--max_speakers", type=int,
        help="Maximum number of speakers (default: 2)",
        default=2
    )
    return parser.parse_args()


def who_speaks_interval(start, end, annotation):
    overlaps = defaultdict(float)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > 0:
            overlaps[speaker] += overlap
    if not overlaps:
        return "Unknown"
    speaker, max_overlap = max(overlaps.items(), key=lambda x: x[1])
    if max_overlap < (end - start) * 0.5:
        return "Unknown"
    return speaker


def main():
    args = parse_args()
    audio_file = args.input_file
    out_stream = open(args.output_file, "w", encoding="utf-8") if args.output_file else sys.stdout

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) openai-whisper transcription
    whisper_model = whisper.load_model(args.model_size, device=device)
    whisper_result = whisper_model.transcribe(
        audio_file,
        word_timestamps=True
    )
    segments = whisper_result.get("segments", [])

    # 2) pyannote diarization
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=True
    ).to(device)
    with ProgressHook() as hook:
        annotation = diar_pipeline(
            audio_file,
            hook=hook,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )

    # 3) Align and output
    for seg in segments:
        spk = who_speaks_interval(seg["start"], seg["end"], annotation)
        out_stream.write(f"[{seg['start']:.1f}s–{seg['end']:.1f}s] ({spk}) {seg['text']}\n")

    if args.output_file:
        out_stream.close()

if __name__ == "__main__":
    main()

