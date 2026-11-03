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


# ---------------------------------------------------------
# FUNCION DE VERIFICACION (Ahora está afuera, limpia y reutilizable)
# ---------------------------------------------------------
def verificar_estado_espeak():
    # Obtenemos la URL base (ej: https://tu-espeak-server.onrender.com)
    base_espeak_url = ESPEAK_SERVICE_URL.rsplit("/", 1)[0]
    
    try:
        # Hacemos un GET rápido con un timeout corto
        response = requests.get(base_espeak_url, timeout=5)
        
        # 1. Evaluamos que sea 200 OK
        # 2. Evaluamos que la respuesta sea texto plano y NO una página HTML de Render (<html...)
        if response.status_code == 200 and "<html" not in response.text.lower():
            print(f"Servidor eSpeak operativo. Status Code: {response.status_code}")
            return True
            
        return False
            
    except requests.exceptions.RequestException:
        # Si hay timeout, error de red o el servidor está dormido
        return False


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

        # 1. Capturar el parámetro 'modo' enviado por el cliente (por defecto 'normal')
        modo = request.form.get("modo", "normal")
        print(f"Modo de respuesta solicitado: {modo}")

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

        # 4. LLM: Qwen
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

        # --- ARCHIVO 1: ai.txt ---
        with open(AI_FILE, "w", encoding="utf-8") as f:
            f.write(ai_sin_emojis)

        # --- ARCHIVO 2: ai_limpio.txt ---
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

        # =========================================================
        # 5. TTS: Bucle de espera inteligente para despertar eSpeak
        # =========================================================
        print("Verificando si el servicio eSpeak está despierto...")
        intentos_restantes = 30  # 30 intentos * 10 segundos = 5 minutos máximo de espera

        # Mientras NO esté operativo Y todavía queden intentos...
        while not verificar_estado_espeak() and intentos_restantes > 0:
            print(f"eSpeak está durmiendo. Reintentando en 10s... (Intentos restantes: {intentos_restantes})")
            time.sleep(10)
            intentos_restantes -= 1  # Restamos un intento correctamente

        # Verificación final tras el bucle
        if not verificar_estado_espeak():
            print("Error: El servidor de eSpeak NO responde tras agotar los intentos.")
            return jsonify({"error": "SERVER de eSpeak NO responde"}), 500

        print("¡El servidor eSpeak está 100% operativo!")

        # Breve pausa de seguridad para asegurar estabilidad
        time.sleep(2)

        print(f"Enviando texto a eSpeak por POST URL: {ESPEAK_SERVICE_URL}")
        print(f"Texto a sintetizar: {texto_limpio_espeak}")

        try:
            # Ahora el POST se ejecuta con el servidor ya activo
            espeak_response = requests.post(
                ESPEAK_SERVICE_URL,
                data=texto_limpio_espeak.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
                timeout=30,
            )
            print(f"Respuesta de eSpeak - Status Code: {espeak_response.status_code}")

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

        if espeak_response.status_code != 200:
            print(f"eSpeak devolvió error body: {espeak_response.text}")
            return (
                jsonify({
                    "error": (
                        f"Fallo al generar el audio en el servicio eSpeak (Status"
                        f" {espeak_response.status_code})"
                    ),
                    "detalle_espeak": espeak_response.text,
                    "stt": stt_text,
                    "ai": ai_sin_emojis,
                }),
                500,
            )

        return espeak_response.content, 200, {"Content-Type": "audio/mpeg"}

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
