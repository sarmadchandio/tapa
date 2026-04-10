"""Whisper-based transcription with speaker attribution."""

import csv


def transcribe_audio(audio_path, whisper_model):
    """Transcribe audio using Whisper with word-level timestamps."""
    result = whisper_model.transcribe(audio_path, word_timestamps=True, language="en")
    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append({"word": w["word"].strip(), "start": w["start"], "end": w["end"]})
    return words


def save_transcription(words, diarization_segments, path):
    """Save word-level transcription with speaker labels to CSV and TXT."""
    rows = []
    for w in words:
        w_mid = (w["start"] + w["end"]) / 2
        speaker = "UNKNOWN"
        for seg in diarization_segments:
            if seg["start"] <= w_mid <= seg["end"]:
                speaker = seg["speaker"]
                break
        rows.append({"speaker": speaker, "word": w["word"],
                     "start": round(w["start"], 4), "end": round(w["end"], 4)})
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["speaker", "word", "start", "end"])
        wr.writeheader()
        wr.writerows(rows)
    txt_path = path.replace(".csv", ".txt")
    with open(txt_path, "w") as f:
        cur_spk, line_words = None, []
        for row in rows:
            if row["speaker"] != cur_spk:
                if line_words:
                    f.write(f"[{cur_spk}] ({line_words[0]['start']:.1f}s - {line_words[-1]['end']:.1f}s)\n")
                    f.write(" ".join(w["word"] for w in line_words) + "\n\n")
                cur_spk = row["speaker"]
                line_words = [row]
            else:
                line_words.append(row)
        if line_words:
            f.write(f"[{cur_spk}] ({line_words[0]['start']:.1f}s - {line_words[-1]['end']:.1f}s)\n")
            f.write(" ".join(w["word"] for w in line_words) + "\n")
