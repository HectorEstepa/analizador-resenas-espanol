"use strict";
const elemento = (id) => document.getElementById(id);
const campo = elemento("texto");
const boton = elemento("analizar");
const resultados = document.querySelector(".resultados");
let funciones = { contexto: false };
function contar() { elemento("contador").textContent = `${campo.value.length} / 2000`; }
campo.addEventListener("input", contar);
document.querySelectorAll("[data-ejemplo]").forEach((botonEjemplo) => {
  botonEjemplo.addEventListener("click", () => { campo.value = botonEjemplo.dataset.ejemplo; contar(); campo.focus(); });
});
fetch("/api/estado").then((r) => { if (!r.ok) throw new Error(); return r.json(); }).then((estado) => {
  funciones = estado;
  elemento("descripcion-modelo").textContent = "El modelo activo distingue sentimiento positivo o negativo.";
  elemento("capacidades").textContent = `Análisis de temas ${estado.contexto ? "activado" : "desactivado"}.`;
}).catch(() => { elemento("capacidades").textContent = "No se pudo consultar el estado del servidor."; });
elemento("formulario").addEventListener("submit", async (evento) => {
  evento.preventDefault();
  boton.disabled = true;
  resultados.setAttribute("aria-busy", "true");
  elemento("error").hidden = true;
  elemento("salida").hidden = true;
  elemento("estado").textContent = "Analizando… La primera carga puede tardar varios minutos.";
  try {
    const respuesta = await fetch("/api/analizar", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ texto: campo.value }) });
    const datos = await respuesta.json();
    if (!respuesta.ok) throw new Error(datos.error || "No se pudo completar el análisis.");
    elemento("sentimiento").textContent = datos.sentimiento.etiqueta;
    elemento("origen-sentimiento").textContent = datos.sentimiento.origen;
    elemento("puntaje").value = datos.sentimiento.puntaje;
    elemento("porcentaje").textContent = `${(datos.sentimiento.puntaje * 100).toFixed(1)} %`;
    elemento("distribucion").replaceChildren();
    Object.entries(datos.sentimiento.probabilidades || {}).forEach(([clase, puntaje]) => {
      const item = document.createElement("li");
      item.textContent = `${clase}: ${(puntaje * 100).toFixed(1)} %`;
      elemento("distribucion").append(item);
    });
    const ordenadas = Object.entries(datos.sentimiento.probabilidades || {}).sort((a, b) => b[1] - a[1]);
    elemento("lectura-puntaje").textContent = ordenadas.length > 1 ? `${ordenadas[0][0]} tiene el puntaje más alto (${(ordenadas[0][1] * 100).toFixed(1)} %), seguido de ${ordenadas[1][0]} (${(ordenadas[1][1] * 100).toFixed(1)} %).` : "Este puntaje corresponde a la clase elegida.";
    elemento("intencion").textContent = datos.intencion ? datos.intencion.etiqueta : (funciones.contexto ? "No disponible en este análisis" : "Función opcional desactivada");
    elemento("origen-intencion").textContent = datos.intencion ? `${datos.intencion.origen}. Puntaje: ${(datos.intencion.puntaje * 100).toFixed(1)} %.` : (funciones.contexto ? "El análisis de temas no está disponible en este momento." : "El análisis de temas no está activado en esta sesión.");
    elemento("advertencia").textContent = [datos.advertencia, ...(datos.avisos || [])].join(" ");
    elemento("salida").hidden = false;
    elemento("estado").textContent = "Análisis completado.";
  } catch (error) {
    elemento("estado").textContent = "El análisis no se completó.";
    elemento("error").textContent = error.message || "No se pudo conectar con el servidor.";
    elemento("error").hidden = false;
  } finally {
    boton.disabled = false;
    resultados.setAttribute("aria-busy", "false");
  }
});

const porcentaje = (valor) => `${(valor * 100).toFixed(1)} %`;
function filaTabla(valores, primeraCabecera = true) {
  const fila = document.createElement("tr");
  valores.forEach((valor, indice) => {
    const celda = document.createElement(primeraCabecera && indice === 0 ? "th" : "td");
    if (primeraCabecera && indice === 0) celda.scope = "row";
    celda.textContent = String(valor);
    fila.append(celda);
  });
  return fila;
}
fetch("/api/metricas").then((r) => { if (!r.ok) throw new Error(); return r.json(); }).then((datos) => {
  if (!datos.disponible) { elemento("estado-metricas").textContent = datos.motivo; return; }
  const reporte = datos.reporte;
  const prueba = reporte.test;
  elemento("metrica-accuracy").textContent = porcentaje(prueba.accuracy);
  elemento("metrica-f1").textContent = prueba.f1_macro.toFixed(3);
  elemento("metrica-n").textContent = prueba.n.toLocaleString("es-CL");
  elemento("comparacion").textContent = `Comparación: predecir siempre la clase mayoritaria obtiene ${porcentaje(reporte.baseline_mayoria.accuracy)} de accuracy y ${reporte.baseline_mayoria.f1_macro.toFixed(3)} de F1 macro.`;
  reporte.clases.forEach((clase) => {
    const valores = prueba.por_clase[clase];
    elemento("metricas-clases").append(filaTabla([clase, porcentaje(valores.precision), porcentaje(valores.recall), valores["f1-score"].toFixed(3), valores.support]));
  });
  const encabezado = document.createElement("tr");
  ["Real / predicho", ...reporte.clases].forEach((valor) => {
    const celda = document.createElement("th"); celda.scope = "col"; celda.textContent = valor; encabezado.append(celda);
  });
  elemento("confusion-cabecera").append(encabezado);
  prueba.matriz_confusion.forEach((fila, i) => elemento("confusion-cuerpo").append(filaTabla([reporte.clases[i], ...fila])));
  const nombres = { "origen:resenas_originales": "Reseñas originales", "origen:sintetico_ia": "Ejemplos sintéticos de IA", modismos_sinteticos: "Modismos sintéticos" };
  Object.entries(prueba.subconjuntos).forEach(([nombre, valores]) => elemento("metricas-grupos").append(filaTabla([nombres[nombre] || nombre, valores.n, porcentaje(valores.accuracy), valores.f1_macro.toFixed(3)])));
  elemento("limites-metricas").textContent = reporte.limitaciones.join(" ") + " En subconjuntos sin ambas clases, F1 macro mantiene las dos etiquetas y asigna cero a clases ausentes.";
  elemento("estado-metricas").textContent = "Prueba separada del entrenamiento y de la validación. Estos resultados no evalúan el modelo opcional de intención.";
  elemento("metricas").hidden = false;
}).catch(() => { elemento("estado-metricas").textContent = "No se pudo cargar el informe de evaluación."; });
