"""Small, self-contained PDF export for a dashboard prediction summary."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _value(prediction: Mapping[str, Any], key: str, default: Any = "-") -> Any:
    value = prediction.get(key, default)
    return default if value is None else value


def build_prediction_report_pdf(
    prediction: Mapping[str, Any], output_path: str | Path | None = None
) -> bytes:
    """Build a one-page Spanish PDF; optionally save it and always return bytes.

    The report is deliberately grounded in the supplied prediction mapping. It
    never invents vehicle causes or represents the score as mechanical
    reliability.
    """

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Resumen AutoReliability",
        author="AutoReliability",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#173E8C"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4B5563"),
        alignment=TA_CENTER,
    )
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#1F2937"),
    )

    make = str(_value(prediction, "marca", "Vehiculo")).title()
    model = str(_value(prediction, "modelo", "consultado")).title()
    year = _value(prediction, "ano_fabricacion", "")
    score = float(prediction["prediccion_indice_100"])
    if not isfinite(score) or not 0 <= score <= 100:
        raise ValueError("El informe requiere un índice válido de 0 a 100.")
    try:
        mae = float(prediction.get("mae"))
        mae_display = f"{mae:.1f} puntos" if isfinite(mae) and mae >= 0 else "No disponible"
    except (ValueError, TypeError):
        mae_display = "No disponible"
    category = str(_value(prediction, "categoria_vehiculo", "Sin categoria"))
    history = _value(prediction, "hist_fiabilidad_marca", "-")
    try:
        history_display = f"{float(history):.2f} pts"
    except (TypeError, ValueError):
        history_display = str(history)
    explanation = escape(str(_value(prediction, "explicacion_factores", "No disponible.")))
    demo_source = bool(_value(prediction, "fuente_demo", False))

    story = [
        Paragraph("AutoReliability", title),
        Paragraph("Resumen de estimacion basada en recalls NHTSA", subtitle),
        Spacer(1, 0.35 * cm),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB")),
        Spacer(1, 0.35 * cm),
        Paragraph(f"<b>Vehiculo analizado:</b> {escape(make)} {escape(model)} - {escape(str(year))}", body),
        Spacer(1, 0.22 * cm),
    ]
    metrics = [
        ["Indice estimado", "MAE temporal", "Categoria", "Historial previo"],
        [f"{score:.1f} / 100", mae_display, Paragraph(escape(category), subtitle), history_display],
    ]
    table = Table(metrics, colWidths=[4.15 * cm, 4.15 * cm, 4.15 * cm, 4.15 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#173E8C")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 0.18 * cm),
            Paragraph("El MAE es el error absoluto medio en la evaluación temporal; no es un intervalo de confianza individual.", body),
            Spacer(1, 0.45 * cm),
            *(
                [
                    Paragraph(
                        "<b>Modo demostración:</b> este informe usa datos sintéticos para validar el producto. "
                        "No representa evidencia real de NHTSA.",
                        body,
                    ),
                    Spacer(1, 0.22 * cm),
                ]
                if demo_source
                else []
            ),
            Paragraph("<b>Interpretacion de los factores</b>", body),
            Paragraph(explanation, body),
            Spacer(1, 0.2 * cm),
            Paragraph(escape(str(_value(prediction, "mensaje", ""))), body),
            Paragraph(f"<b>Modelo:</b> {escape(str(_value(prediction, 'modelo_usado')))}. "
                      f"<b>Fecha:</b> {escape(str(_value(prediction, 'fecha_ejecucion')))}", body),
            Spacer(1, 0.4 * cm),
            Paragraph(
                "<b>Limitacion importante:</b> este indice es un proxy construido a partir "
                "de recalls de seguridad de NHTSA durante una ventana de observacion comun. "
                "No equivale a la fiabilidad mecanica general ni garantiza la ausencia de averias.",
                body,
            ),
            Spacer(1, 0.24 * cm),
            Paragraph(
                "Un valor mas alto indica menor propension historica estimada a recalls. "
                "La estimacion debe utilizarse como apoyo a la decision, no como certeza.",
                body,
            ),
        ]
    )
    document.build(story)
    contents = buffer.getvalue()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
    return contents
