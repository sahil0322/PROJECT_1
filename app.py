from flask import Flask, render_template, request, jsonify
import whisper
import torch
import soundfile as sf
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
import google.generativeai as genai


app = Flask(__name__, static_folder='templates', static_url_path='')

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
HF_TOKEN = os.getenv("HF_TOKEN")


print("Loading AI Models into memory... (This takes a moment)")
whisper_model = whisper.load_model("base")
diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_audio():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['audio_file']
    

    tmp_path = "temp_audio.wav"
    file.save(tmp_path)

    try:
        print("Transcribing audio...")
        transcript = whisper_model.transcribe(tmp_path)
        segments = transcript["segments"]
        
        print("Identifying speakers...")
        audio_array, sample_rate = sf.read(tmp_path)
        if len(audio_array.shape) == 1:
            audio_array = audio_array.reshape(1, -1)
        else:
            audio_array = audio_array.T
        waveform = torch.tensor(audio_array, dtype=torch.float32)
        
        diarization = diarize_model({"waveform": waveform, "sample_rate": sample_rate}).speaker_diarization
        
        print("Merging text and formatting for UI...")
        raw_transcript_for_gemini = ""
        html_transcript_for_ui = ""
        
        speaker_map = {}
        speaker_counter = 1

        for segment in segments:
            midpoint = (segment["start"] + segment["end"]) / 2
            current_speaker_raw = "UNKNOWN"
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= midpoint <= turn.end:
                    current_speaker_raw = speaker
                    break
            
            if current_speaker_raw not in speaker_map:
                speaker_map[current_speaker_raw] = f"Speaker {speaker_counter:02d}"
                speaker_counter += 1
            
            clean_speaker_name = speaker_map[current_speaker_raw]
            css_class = "speaker-1" if clean_speaker_name == "Speaker 01" else "speaker-2"
           
            minutes = int(segment["start"] // 60)
            seconds = int(segment["start"] % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            text = segment['text'].strip()
         
            raw_transcript_for_gemini += f"{clean_speaker_name}: {text}\n"
           
            html_transcript_for_ui += f"""
            <div class="transcript-line">
                <div class="speaker-meta">
                    <span class="badge {css_class}">{clean_speaker_name}</span>
                    <span class="timestamp">{timestamp}</span>
                </div>
                <div class="transcript-text">
                    {text}
                </div>
            </div>
            """

        print("Generating summary with Gemini...")
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert executive assistant. Read the meeting transcript and generate structured notes.
        You MUST output ONLY valid JSON matching this exact structure:
        {{
            "insights": ["Insight 1 here", "Insight 2 here"],
            "decisions": ["Decision 1 here", "Decision 2 here"],
            "actions": ["Task 1 here", "Task 2 here"]
        }}
        Do not include markdown formatting like ```json.

        Transcript:
        {raw_transcript_for_gemini}
        """
        
        
        summary_json_string = gemini_model.generate_content(prompt).text.strip()
        
    
        if summary_json_string.startswith("```json"):
             summary_json_string = summary_json_string[7:-3].strip()
        elif summary_json_string.startswith("```"):
             summary_json_string = summary_json_string[3:-3].strip()

       
        return jsonify({
            "summary_json": summary_json_string,
            "transcript_html": html_transcript_for_ui
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)