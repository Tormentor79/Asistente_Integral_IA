import os
import requests
from flask import Flask, jsonify, request, send_file
from groq import Groq

app = Flask(__name__)

# Inicializamos el cliente de Groq usando la variable de entorno que configuraremos en Render
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# URL de tu servicio eSpeak actual en Render (para convertir texto a audio)
ESPEAK_SERVICE_URL = os.environ.get(
    "ESPEAK_SERVICE_URL", "https://tu-espeak-server.onrender.com/tts"
)


@app.route("/", methods=["GET"])
def home():
  return (
      "¡Servidor Integral de Voz funcionando correctamente!",
      200,
  )


@app.route("/procesar-voz", methods=["POST"])
def procesar_voz():
  try:
    # 1. Verificar si llegó un archivo de audio en la petición
    if "audio" not in request.files:
      return jsonify({"error": "No se encontró el archivo de audio"}), 400

    audio_file = request.files["audio"]

    # Guardamos temporalmente el audio recibido para enviarlo a Groq
    temp_audio_path = "temp_input.wav"
    audio_file.save(temp_audio_path)

    # 2. STT: Enviar audio a Groq (Whisper)
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

    # 3. LLM: Enviar el texto reconocido al modelo de chat de Groq
    print("Procesando respuesta con la IA...")
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": stt_text,
        }],
        model="qwen/qwen3.8-27b",  # O el modelo que estés usando
    )
    ai_response_text = chat_completion.choices[0].message.content
    print(f"Respuesta de la IA: {ai_response_text}")

    # Limpiamos el archivo temporal de audio del disco
    if os.path.exists(temp_audio_path):
      os.remove(temp_audio_path)

    # 4. TTS: Mandar el texto de la IA a tu servicio eSpeak para generar el audio de salida
    # (Opcional: puedes mandar la respuesta como audio directo o empaquetarla)
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

    # Revisamos las cabeceras (headers) si el ESP8266 pide guardar los .txt en la SD
    # Si el cliente pide los textos, podemos devolverlos de forma limpia o mantenerlos en endpoints dedicados.
    # Por ahora, devolvemos el binario de audio de eSpeak directamente:
    return espeak_response.content, 200, {"Content-Type": "audio/mpeg"}

  except Exception as e:
    print(f"Error crítico en el servidor: {str(e)}")
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
