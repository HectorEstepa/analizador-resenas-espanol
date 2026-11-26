"""Interfaz web local. No registra ni guarda el texto de las reseñas."""
import argparse
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from resenas.analisis import Analizador
from resenas.modelos import Modelos

BASE = Path(__file__).resolve().parent
MODELO_PREDETERMINADO = BASE / "modelo_publicado"


def crear_app(modelos=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    analizador = Analizador(modelos if modelos is not None else Modelos(MODELO_PREDETERMINADO))

    @app.get("/")
    def inicio():
        return render_template("index.html")

    @app.get("/api/estado")
    def estado():
        informacion = analizador.modelos.informacion() if hasattr(analizador.modelos, "informacion") else {"motor": "prueba", "clases": ["NEGATIVO", "POSITIVO"]}
        return jsonify({"contexto": analizador.modelos.tiene_contexto, "modelo": informacion})

    @app.get("/api/metricas")
    def metricas():
        return jsonify(analizador.modelos.metricas() if hasattr(analizador.modelos, "metricas") else {"disponible": False, "motivo": "Sin informe de evaluación."})

    @app.post("/api/analizar")
    def analizar():
        if not request.is_json:
            return jsonify(error="Envía un objeto JSON con el campo texto."), 415
        # Defensa adicional para el servidor local: el navegador debe llamar desde su propio origen.
        origen = request.headers.get("Origin")
        if origen and origen.rstrip("/") != request.host_url.rstrip("/"):
            return jsonify(error="Origen no permitido."), 403
        try:
            datos = request.get_json()
        except BadRequest:
            return jsonify(error="El JSON no es válido."), 400
        if not isinstance(datos, dict):
            return jsonify(error="Envía un objeto JSON con el campo texto."), 400
        try:
            texto = datos.get("texto")
            # Validación antes de cargar modelos: errores de dependencias no se confunden con entrada inválida.
            from resenas.analisis import validar_texto
            validar_texto(texto)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        try:
            return jsonify(analizador.analizar(texto))
        except Exception:
            logging.getLogger(__name__).exception("No se pudo completar la inferencia")
            return jsonify(error="No se pudo analizar. Comprueba el modelo local, las dependencias y la conexión si activaste modelos externos. Consulta la terminal para más detalles."), 503

    @app.errorhandler(RequestEntityTooLarge)
    def demasiado_grande(error):
        return jsonify(error="La solicitud es demasiado grande."), 413

    @app.after_request
    def cabeceras(respuesta):
        respuesta.headers["X-Content-Type-Options"] = "nosniff"
        respuesta.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        respuesta.headers["Cache-Control"] = "no-store"
        return respuesta

    return app


def main():
    parser = argparse.ArgumentParser(description="Analizador web de reseñas en español.")
    parser.add_argument("--modelo", type=Path, default=MODELO_PREDETERMINADO)
    parser.add_argument("--contexto", action="store_true", help="Activa el modelo externo de temas (requiere PyTorch).")
    parser.add_argument("--puerto", type=int, default=5000)
    args = parser.parse_args()
    crear_app(Modelos(args.modelo, args.contexto)).run(host="127.0.0.1", port=args.puerto, debug=False)


if __name__ == "__main__":
    main()
