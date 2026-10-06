import gradio as gr
import assemblyai as aai
from deep_translator import GoogleTranslator
from elevenlabs import ElevenLabs, VoiceSettings
import uuid
from pathlib import Path

# ----------- 1️⃣ SETUP API KEYS -----------
aai.settings.api_key = "0800c3bdb758493abbe8b8a2c9f1603b"   # <-- Replace with your AssemblyAI key
ELEVEN_API_KEY = "sk_95fe7ad9a154a121356d1b498c956369596650d8018cbe9d"   # <-- Replace with your ElevenLabs key


# ----------- 2️⃣ MAIN FUNCTION -----------
def voice_to_voice(audio_file):
    transcription_response = audio_transcription(audio_file)

    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    # Translate to multiple languages
    es_translation, tr_translation, ja_translation = text_translation(text)

    # Convert translations to voice
    es_audio_path = text_to_speech(es_translation)
    tr_audio_path = text_to_speech(tr_translation)
    ja_audio_path = text_to_speech(ja_translation)

    return es_audio_path, tr_audio_path, ja_audio_path


# ----------- 3️⃣ AUDIO → TEXT -----------
def audio_transcription(audio_file):
    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)
    return transcription


# ----------- 4️⃣ TEXT → TRANSLATIONS -----------
def text_translation(text):
    es_text = GoogleTranslator(source='en', target='es').translate(text)
    tr_text = GoogleTranslator(source='en', target='tr').translate(text)
    ja_text = GoogleTranslator(source='en', target='ja').translate(text)
    return es_text, tr_text, ja_text


# ----------- 5️⃣ TEXT → SPEECH -----------
def text_to_speech(text):
    client = ElevenLabs(api_key=ELEVEN_API_KEY)

    response = client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        model_id="eleven_multilingual_v2",
        text=text,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
        output_format="mp3_22050_32",
    )

    # Save MP3 output
    output_path = f"{uuid.uuid4()}.mp3"
    with open(output_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"✅ Saved: {output_path}")
    return output_path


# ----------- 6️⃣ GRADIO UI -----------
audio_input = gr.Audio(sources=["microphone"], type="filepath")

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[
        gr.Audio(label="Spanish"),
        gr.Audio(label="Turkish"),
        gr.Audio(label="Japanese")
    ],
    title="🌎 Voice Translator",
    description="Speak and hear your voice translated into Spanish, Turkish, and Japanese."
)

if __name__ == "__main__":
    demo.launch()
