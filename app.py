import os
import re
import time
import requests
from flask import Flask, jsonify, request, send_file
from groq import Groq

app = Flask(__name__)

# URL de tu servicio eSpeak en Render
ESPEAK_SERVICE_URL = os.environ.get(
    "ESPEAK_SERVICE_URL", "https://tu-espeak-server.onrender.com/tts"
)

# Rutas absolutas seguras en la carpeta actual del script
STT_FILE = os.path.join(os.getcwd(), "stt.txt")
AI_FILE = os.path.join(os.getcwd(), "ai.txt")  # Con formato markdown y saltos de línea (sin emojis)
AI_LIMPIO_FILE = os.path.join(os.getcwd(), "ai_limpio.txt")  # Texto plano absoluto para eSpeak


def limpiar_emojis(texto):
  # Patrón para eliminar emojis y símbolos gráficos Unicode
  return re.sub(
      r"[\U00010000-\U0010ffff]|[\u2600-\u27BF]|[\U0001f300-\U0001f9ff]",
      "",
      texto,
  )


@app.route("/", methods=["GET"])
def home():
  return "¡Servidor Integral de IA funcionando correctamente!", 200


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

    # 1. Capturar parámetros dinámicos del formulario multipart enviados por el cliente/ESP8266
    modo = request.form.get("modo", "normal")       # 'corto', 'normal' o 'largo'
    tts_motor = request.form.get("tts", "canopy")   # 'canopy' (por defecto) o 'espeak'
    
    print(f"Modo de respuesta solicitado: {modo}")
    print(f"Motor TTS solicitado: {tts_motor}")

    # 2. Configurar dinámicamente el prompt y los tokens según el modo
    if modo == "corto":
      system_content = (
          "Eres un asistente de voz inteligente. Sé muy conciso y directo, "
          "respondiendo en máximo 1 o 2 oraciones. NO utilices emojis bajo ningún concepto."
      )
      max_tokens_val = 100
    elif modo == "largo":
      system_content = (
          "Eres un asistente de voz inteligente. Puedes explayarte de forma muy "
          "detallada y descriptiva (unas 6 u 8 oraciones). NO utilices emojis bajo ningún concepto. "
          "Puedes usar formato markdown y saltos de línea."
      )
      max_tokens_val = 450
    else:  # Modo normal por defecto
      system_content = (
          "Eres un asistente de voz inteligente. Puedes explayarte "
          "de forma natural y detallada (aproximadamente unas 4 o "
          "5 oraciones). NO utilices emojis bajo ningún concepto. "
          "Puedes usar formato markdown y saltos de línea si lo "
          "consideras útil para estructurar tu respuesta."
      )
      max_tokens_val = 300

    audio_file = request.files["audio"]
    temp_audio_path = "temp_input.wav"
    audio_file.save(temp_audio_path)

    # 3. STT: Whisper
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

    # Guardar STT en stt.txt
    with open(STT_FILE, "w", encoding="utf-8") as f:
      f.write(stt_text)

    # 4. LLM: Qwen (con prompt y tokens dinámicos según el modo)
    print("Procesando respuesta con la IA...")
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": stt_text},
        ],
        model="qwen/qwen3.8-27b",
        max_tokens=max_tokens_val,
    )
    ai_response_raw = chat_completion.choices[0].message.content
    print(f"Respuesta cruda de la IA: {ai_response_raw}")

    # Limpiamos únicamente los emojis para el primer archivo
    ai_sin_emojis = limpiar_emojis(ai_response_raw)

    # --- ARCHIVO 1: ai.txt (Conserva formato markdown y saltos de línea, sin emojis) ---
    with open(AI_FILE, "w", encoding="utf-8") as f:
      f.write(ai_sin_emojis)

    # --- ARCHIVO 2: ai_limpio.txt (Sin emojis, sin markdown, sin saltos de línea ni retornos) ---
    texto_limpio_espeak = re.sub(r"[\*\#\_\-\`]", "", ai_sin_emojis)
    texto_limpio_espeak = texto_limpio_espeak.replace("\r", " ").replace(
        "\n", " "
    )
    texto_limpio_espeak = re.sub(r"\s+", " ", texto_limpio_espeak).strip()

    with open(AI_LIMPIO_FILE, "w", encoding="utf-8") as f:
      f.write(texto_limpio_espeak)

    # Limpiar el audio temporal de entrada
    if os.path.exists(temp_audio_path):
      os.remove(temp_audio_path)

    # 5. TTS: ENRUTAMIENTO INTELIGENTE (Canopy en Groq como principal + eSpeak en Render como respaldo)
    audio_bytes = None

    # --- OPCIÓN A: Intentar con Canopy/Orpheus vía API de Groq ---
    if tts_motor == "canopy":
      print("Generando audio TTS con Canopy (Groq)...")
      try:
        response_tts = client.audio.speech.create(
            model="canopylabs/orpheus-v1",  # Ajustar según el identificador de modelo disponible en tu consola de Groq
            voice="default",
            input=texto_limpio_espeak,
        )
        audio_bytes = response_tts.read()
        print("¡Audio generado con éxito mediante Canopy!")
      except Exception as canopy_err:
        print(f"Aviso: Falló el servicio Canopy ({canopy_err}). Conmutando automáticamente a eSpeak (Respaldo)...")
        tts_motor = "espeak"  # Fuerza la entrada al bloque de respaldo eSpeak

    # --- OPCIÓN B: eSpeak en Render (Respaldo si falló Canopy o si fue solicitado explícitamente) ---
    if tts_motor == "espeak":
      base_espeak_url = ESPEAK_SERVICE_URL.rsplit("/", 1)[0]
      print(f"Despertando eSpeak mediante GET en: {base_espeak_url}")

      try:
        res_get = requests.get(base_espeak_url, timeout=45)
        print(f"Servidor eSpeak despertó. Status Code del GET: {res_get.status_code}")
      except Exception as wakeup_err:
        print(f"Aviso en el despertar GET de eSpeak (continuando de todas formas): {wakeup_err}")

      time.sleep(2)

      print(f"Enviando texto a eSpeak por POST URL: {ESPEAK_SERVICE_URL}")
      print(f"Texto a sintetizar: {texto_limpio_espeak}")

      try:
        espeak_response = requests.post(
            ESPEAK_SERVICE_URL,
            data=texto_limpio_espeak.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        print(f"Respuesta de eSpeak - Status Code: {espeak_response.status_code}")

        if espeak_response.status_code == 200:
          audio_bytes = espeak_response.content
          print("¡Audio generado con éxito mediante eSpeak!")
        else:
          print(f"eSpeak devolvió error body: {espeak_response.text}")
          return (
              jsonify({
                  "error": f"Fallo al generar el audio en eSpeak (Status {espeak_response.status_code})",
                  "detalle_espeak": espeak_response.text,
                  "stt": stt_text,
                  "ai": ai_sin_emojis,
              }),
              500,
          )

      except Exception as e:
        print(f"Error de conexión con eSpeak en el POST: {str(e)}")
        return (
            jsonify({
                "error": f"No se pudo conectar con el servicio eSpeak: {str(e)}",
                "stt": stt_text,
                "ai": ai_sin_emojis,
            }),
            500,
        )

    # Retornar la respuesta de audio al cliente/ESP8266
    if audio_bytes:
      return audio_bytes, 200, {"Content-Type": "audio/mpeg"}
    else:
      return jsonify({"error": "No se pudo sintetizar el audio con ningún motor TTS disponible"}), 500

  except Exception as e:
    print(f"Error crítico en el servidor: {str(e)}")
    return jsonify({"error": str(e)}), 500


# --- ENDPOINTS DE DESCARGA PARA LA SD DEL ESP8266 ---


@app.route("/logs/stt", methods=["GET"])
def descargar_stt():
  if os.path.exists(STT_FILE):
    return send_file(STT_FILE, mimetype="text/plain", as_attachment=True)
  return "Archivo STT no encontrado", 404


@app.route("/logs/ai", methods=["GET"])
def descargar_ai():
  if os.path.exists(AI_FILE):
    return send_file(AI_FILE, mimetype="text/plain", as_attachment=True)
  return "Archivo AI con formato no encontrado", 404


@app.route("/logs/ai-limpio", methods=["GET"])
def descargar_ai_limpio():
  if os.path.exists(AI_LIMPIO_FILE):
    return send_file(AI_LIMPIO_FILE, mimetype="text/plain", as_attachment=True)
  return "Archivo AI limpio no encontrado", 404


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
