import os
import requests
from flask import Flask, jsonify, request
from groq import Groq

app = Flask(__name__)

# URL de tu servicio eSpeak actual en Render
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
    # Inicializamos el cliente de Groq aquí adentro de forma segura
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
      return (
          jsonify({"error": "GROQ_API_KEY no configurada en las variables de Render"}),
          500,
      )

    client = Groq(api_key=groq_api_key)

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
        model="qwen/qwen3.8-27b",
    )
    ai_response_text = chat_completion.choices[0].message.content
    print(f"Respuesta de la IA: {ai_response_text}")

    # Limpiamos el archivo temporal de audio del disco
    if os.path.exists(temp_audio_path):
      os.remove(temp_audio_path)

    # 4. TTS: Limpiar formato markdown (asteriscos, viñetas) manteniendo acentos y puntuación
    import re

    texto_para_voz = ai_response_text

    # Quitamos símbolos molestos de markdown (*, #, -, _, `)
    texto_para_voz = re.sub(r"[\*\#\_\-\`]", "", texto_para_voz)

    # Reemplazamos saltos de línea por espacios para que eSpeak no haga silencios raros
    texto_para_voz = texto_para_voz.replace("\n", " ")

    # Limpiamos espacios dobles generados por los reemplazos
    texto_para_voz = re.sub(r"\s+", " ", texto_para_voz).strip()

    print(f"Texto limpio para eSpeak: {texto_para_voz}")

    espeak_response = requests.post(
        ESPEAK_SERVICE_URL,
        data=texto_para_voz.encode("utf-8"),
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


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
