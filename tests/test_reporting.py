from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from auto_reliability.reporting import build_prediction_report_pdf


def test_pdf_report_is_readable_and_preserves_the_proxy_disclaimer(tmp_path) -> None:
    destination = tmp_path / "prediction.pdf"
    payload = build_prediction_report_pdf(
        {
            "marca": "ford",
            "modelo": "explorer",
            "ano_fabricacion": 2023,
            "prediccion_indice_100": 72.4,
            "mae": 6.2,
            "categoria_vehiculo": "SUV",
            "hist_fiabilidad_marca": 1.8,
            "explicacion_factores": "La potencia y el historial previo aportan el mayor contexto.",
            "fuente_demo": True,
        },
        destination,
    )
    assert payload.startswith(b"%PDF")
    reader = PdfReader(destination)
    text = reader.pages[0].extract_text()
    assert "Modo demostración" in text
    assert "proxy construido" in text


def test_pdf_does_not_invent_missing_mae():
    payload = build_prediction_report_pdf({"prediccion_indice_100": 50, "mae": None})
    text = PdfReader(BytesIO(payload)).pages[0].extract_text()
    assert "No disponible" in text
    assert "no es un intervalo" in text
    assert "+/- 0.0" not in text


def test_pdf_preserves_model_version_and_official_provenance():
    payload = build_prediction_report_pdf({"prediccion_indice_100": 50, "version_modelo": "model-test-abc",
        "version_datos": "data-test-def", "version_escala": "scale-test-ghi",
        "evidencia_oficial": {"status": "unavailable", "count": None}})
    text = " ".join(page.extract_text() for page in PdfReader(BytesIO(payload)).pages)
    assert all(value in text for value in ("model-test-abc", "data-test-def", "scale-test-ghi", "unavailable"))
