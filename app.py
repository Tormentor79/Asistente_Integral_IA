import os
import requests
from flask import Flask, jsonify, request, send_file
from groq import Groq

app = Flask(__name__)

ESPEAK_SERVICE_URL = os.environ.get(
    "ESPEAK_SERVICE_URL", "https://tu-espeak-server.onrender.com/tts"
)

# Archivos temporales donde guardamos el último intercambio
STT_FILE = "stt.txt"
AI_FILE = "ai.txt"


@app.route("/", methods=["GET"])
def home():
  return "¡Servidor Integral de Voz funcionando correctamente!", 200


@app.route("/procesar-voz", methods=["POST"])
def procesar_voz():
  try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
      return (
          jsonify({"error": "GROQ_API_KEY no configurada en Render"}),
          500,
      )

    client = Groq(api_key=groq_api_key)

    if "audio" not in request.files:
      return jsonify({"error": "No se encontró el archivo de audio"}), 400

    audio_file = request.files["audio"]
    temp_audio_path = "temp_input.wav"
    audio_file.save(temp_audio_path)

    # 1. STT: Whisper
    print("Transcribiendo audio con Whisper...")
    with open(temp_audio_path, "rb") as file:
      transcription = client.audio.transcriptions.create(
          file=(temp_audio_path, file.read()),
          model="whisper-large-v3",
          response_format="text",
      )
    stt_text = (
        transcription if isinstance(transcription, str) else transcription.text
    )
    print(f"Texto reconocido (STT): {stt_text}")

    # Guardamos el STT en stt.txt
    with open(STT_FILE, "w", encoding="utf-8") as f:
      f.write(stt_text)

    # 2. LLM: Qwen
    print("Procesando respuesta con la IA...")
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": stt_text}],
        model="qwen/qwen3.8-27b",
    )
    ai_response_text = chat_completion.choices[0].message.content
    print(f"Respuesta de la IA: {ai_response_text}")

    # Guardamos la respuesta en ai.txt
    with open(AI_FILE, "w", encoding="utf-8") as f:
      f.write(ai_response_text)

    # Limpiamos el audio temporal
    if os.path.exists(temp_audio_path):
      os.remove(temp_audio_path)

    # 3. TTS: Mandar el contenido de ai.txt a eSpeak por POST (datos crudos)
    espeak_response = requests.post(
        ESPEAK_SERVICE_URL,
        data=ai_response_text.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
    )

    if espeak_response.status_code != 200:
      return (
          jsonify({
              "error": "Fallo al generar el audio en el servicio eSpeak",
              "stt": stt_text,
              "ai": ai_response_text,
          }),
          500,
      )

    return espeak_response.content, 200, {"Content-Type": "audio/mpeg"}

  except Exception as e:
    print(f"Error crítico en el servidor: {str(e)}")
    return jsonify({"error": str(e)}), 500


# Endpoints para que el ESP8266 descargue los archivos de texto directamente a la SD
@app.route("/logs/stt", methods=["GET"])
def descargar_stt():
  if os.path.exists(STT_FILE):
    return send_file(STT_FILE, mimetype="text/plain", as_attachment=True)
  return "Archivo no encontrado", 404


@app.route("/logs/ai", methods=["GET"])
def descargar_ai():
  if os.path.exists(AI_FILE):
    return send_file(AI_FILE, mimetype="text/plain", as_attachment=True)
  return "Archivo no encontrado", 404


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
