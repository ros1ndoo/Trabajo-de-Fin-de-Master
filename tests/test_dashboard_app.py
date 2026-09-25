"""Prueba el ciclo real de interacción Streamlit sin red ni artefactos externos."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from auto_reliability.dashboard import PLACEHOLDER
from auto_reliability.service import InsufficientEvidenceError


@dataclass
class DashboardServiceStub:
    demo: bool = False
    error: bool = False
    live_error: bool = False
    mae: float | None = 6.5
    score: float = 62.5
    live_count: int = 1
    ingestion_warnings: tuple[str, ...] = ()
    inventory_alignment: dict | None = None
    exclusion_review: dict | None = None
    live_calls: list[tuple[str, str, int]] = field(default_factory=list)

    def status(self):
        return {
            "demo_mode": self.demo, "message": "Datos de prueba controlados.",
            "model_available": True, "technical_year_max": 2017,
            "ingestion_warnings": self.ingestion_warnings,
            "inventory_alignment": self.inventory_alignment,
            "exclusion_review": self.exclusion_review,
        }

    def load_catalog(self):
        return pd.DataFrame([
            {"marca": make, "modelo": model, "ano_fabricacion": year,
             "categoria_vehiculo": category, "mediana_cilindros": 6,
             "mediana_cv": 250, "hist_fiabilidad_marca": 1.2,
             "indice_fiabilidad_100": 50, "fuente_demo": self.demo}
            for make, model, year, category in [
                ("ford", "explorer", 2010, "suv"),
                ("ford", "explorer", 2016, "suv"),
                ("ford", "explorer", 2017, "suv"),
                ("chevrolet", "malibu", 2017, "sedan"),
            ]
        ])

    def predict(self, marca, modelo, ano_fabricacion):
        if self.error:
            raise InsufficientEvidenceError("Controlled missing-history example")
        return {
            "marca": marca, "modelo": modelo, "ano_fabricacion": ano_fabricacion,
            "prediccion_indice_100": self.score, "mae": self.mae, "fuente_demo": self.demo,
            "modelo_usado": "Ridge", "mensaje": "Consulta retrospectiva: no es una predicción realizada al lanzamiento.",
            "factores": [{"feature": "mediana_cv", "value": 250, "contribution": 0.0}],
        }

    def real_recalls_catalog(self):
        return self.load_catalog()

    def real_recalls(self, marca, modelo, year):
        self.live_calls.append((marca, modelo, year))
        if self.live_error:
            raise ConnectionError("Controlled network failure")
        return {
            "records": [{"NHTSACampaignNumber": "26V000001", "Component": "BRAKES",
                         "Summary": "Official fixture recall", "Remedy": "Contact the manufacturer"}]
            if self.live_count else [],
            "count": self.live_count, "fetched_at": "2026-09-18T00:00:00Z", "from_cache": True,
            "source": "NHTSA", "notice": "No identifica si un VIN particular está afectado.",
        }


def _app_script(service):
    from auto_reliability.dashboard import run_dashboard

    run_dashboard(service)


def _start(service=None):
    service = service or DashboardServiceStub()
    app = AppTest.from_function(_app_script, args=(service,), default_timeout=15).run()
    assert not app.exception
    return app


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _select(app, year=2017):
    app.selectbox(key="ar_marca").select("ford").run()
    app.selectbox(key="ar_modelo").select("explorer").run()
    app.selectbox(key="ar_ano").select(year).run()
    return app


def _estimate(app):
    _button(app, "Estimar propensión a recalls").click().run()
    assert not app.exception
    return app


def _select_official(app):
    assert not app.text_input
    assert app.selectbox(key="ar_real_model").disabled
    app.selectbox(key="ar_real_make").select("ford").run()
    app.selectbox(key="ar_real_model").select("explorer").run()
    app.selectbox(key="ar_real_year").select(2017).run()
    return app


def test_internal_reports_are_hidden_without_hiding_coverage_warnings():
    app = _start(DashboardServiceStub(
        inventory_alignment={
            "variants": 36431, "source_names": 22140, "linked_source_names": 4570,
            "epa_snapshot_date": "2026-09-18", "policy_version": "epa-nhtsa-names-v1",
        },
        exclusion_review={
            "excluded_rows": 854,
            "independent_identity_counts": {"exact_independent_identity": 380},
            "query_counts": {"campaigns_returned_review_required": 40, "query_failed": 340},
        },
        ingestion_warnings=("Muestra limitada; posible sesgo de selección.",),
    ))
    labels = {item.label for item in app.expander}
    assert "Inventario complementario EPA/NHTSA · en validación" not in labels
    assert "Revisión de exclusiones · evidencia independiente" not in labels
    assert not any("4570 de 22140" in item.value for item in app.markdown)
    assert not any("854 excluidos" in item.value for item in app.markdown)
    assert not any("no amplía todavía el predictor" in item.value for item in app.info)
    assert any("hasta 2017" in item.value for item in app.warning)
    assert any("sesgo de selección" in item.value for item in app.warning)
    assert any("Datos de prueba controlados" in item.value for item in app.caption)
    assert not app.exception


def test_trained_baseline_remains_visible_when_official_api_is_unavailable():
    class BaselineService(DashboardServiceStub):
        def predict_on_demand(self, marca, modelo, ano_fabricacion):
            result = super().predict(marca, modelo, ano_fabricacion)
            result.update({
                "es_baseline": True, "modelo_usado": "Baseline histórico",
                "mensaje": "Estimación de grupo: no distingue modelos dentro de la misma marca y segmento.",
                "evidencia_oficial": {"status": "unavailable", "count": None},
            })
            return result

    app = _estimate(_select(_start(BaselineService())))
    assert app.session_state["ar_current_result"]["prediccion_indice_100"] == 62.5
    assert any("no se ha supuesto" in item.value for item in app.warning)
    assert any("no distingue modelos" in item.value for item in app.warning)
    assert not app.exception


def test_unverified_empty_official_response_is_not_presented_as_zero_risk():
    class UnverifiedService(DashboardServiceStub):
        def predict_on_demand(self, marca, modelo, ano_fabricacion):
            result = super().predict(marca, modelo, ano_fabricacion)
            result["evidencia_oficial"] = {
                "status": "queried", "count": 0,
                "identity_status": "technical_name_unverified_in_nhtsa",
                "source": "NHTSA", "fetched_at": "2026-09-19",
            }
            return result

    app = _estimate(_select(_start(UnverifiedService())))
    assert any("no certifica ausencia" in item.value for item in app.warning)
    assert app.session_state["ar_current_result"]["prediccion_indice_100"] == 62.5
    assert not app.exception


def test_internal_report_errors_are_hidden_without_breaking_predictor():
    app = _estimate(_select(_start(DashboardServiceStub(
        inventory_alignment={"error": "Inventario no válido"},
        exclusion_review={"error": "Revisión no válida"},
        ingestion_warnings=("Muestra limitada; posible sesgo de selección.",),
    ))))
    labels = {item.label for item in app.expander}
    assert "Inventario complementario EPA/NHTSA · en validación" not in labels
    assert "Revisión de exclusiones · evidencia independiente" not in labels
    assert not any("Inventario no válido" in item.value for item in app.warning)
    assert not any("Revisión no válida" in item.value for item in app.warning)
    assert any("hasta 2017" in item.value for item in app.warning)
    assert any("sesgo de selección" in item.value for item in app.warning)
    assert app.session_state["ar_current_result"]["prediccion_indice_100"] == 62.5
    assert not app.exception


@pytest.mark.parametrize("score", [30.0, 62.5, 85.0])
def test_result_omits_score_bands_but_keeps_score_and_interpretation(score):
    app = _estimate(_select(_start(DashboardServiceStub(score=score))))
    rendered_text = "\n".join(
        str(item.value)
        for elements in (app.markdown, app.caption, app.info, app.warning, app.success)
        for item in elements
    )
    assert "tramo" not in rendered_text.casefold()
    assert "45/70" not in rendered_text
    assert "convenciones visuales" not in rendered_text
    assert "Cómo interpretar este resultado" in rendered_text
    assert "No mide directamente la fiabilidad mecánica general" in rendered_text
    assert f"{score:.1f}" in rendered_text
    assert app.session_state["ar_current_result"]["prediccion_indice_100"] == score


def test_cascading_selectors_estimation_comparison_and_reset():
    app = _start()
    assert app.selectbox(key="ar_modelo").disabled
    assert app.selectbox(key="ar_ano").disabled
    assert any("hasta 2017" in item.value for item in app.warning)
    _estimate(app)
    assert any("Selecciona marca" in item.value for item in app.error)
    _estimate(_select(app))
    assert app.session_state["ar_current_result"]["prediccion_indice_100"] == 62.5
    assert any("retrospectiva" in item.value for item in app.info)
    _button(app, "Añadir a comparación").click().run()
    assert len(app.session_state["ar_comparison"]) == 1
    app.selectbox(key="ar_ano").select(2016).run()
    assert "ar_current_result" not in app.session_state
    _estimate(app)
    _button(app, "Añadir a comparación").click().run()
    assert len(app.session_state["ar_comparison"]) == 2
    assert any("Comparación guardada (2/2)" == item.label for item in app.expander)
    app.selectbox(key="ar_marca").select("chevrolet").run()
    assert app.selectbox(key="ar_modelo").value == PLACEHOLDER
    assert app.selectbox(key="ar_ano").value == PLACEHOLDER
    assert "ar_current_result" not in app.session_state
    _button(app, "Reiniciar").click().run()
    assert app.selectbox(key="ar_marca").value == PLACEHOLDER
    assert "ar_comparison" not in app.session_state
    assert not app.exception


def test_demo_is_visible_and_missing_mae_is_not_zero():
    app = _estimate(_select(_start(DashboardServiceStub(demo=True, mae=None))))
    assert any("MODO DEMOSTRACIÓN" in element.value for element in app.markdown)
    assert any("datos sintéticos" in element.value for element in app.error)
    assert next(metric.value for metric in app.metric if metric.label == "Error esperado (MAE)") == "No disponible"


def test_coverage_does_not_hide_status_or_ingestion_limitations():
    app = _start(DashboardServiceStub(ingestion_warnings=("Muestra limitada; posible sesgo de selección.",)))
    assert any("hasta 2017" in item.value for item in app.warning)
    assert any("sesgo de selección" in item.value for item in app.warning)
    assert any("Datos de prueba controlados" in item.value for item in app.caption)


def test_missing_evidence_reports_error_without_fabricated_score():
    app = _estimate(_select(_start(DashboardServiceStub(error=True))))
    assert any("No se puede emitir una estimación" in item.value for item in app.error)
    assert "ar_current_result" not in app.session_state


def test_official_recalls_require_submit_and_persist_on_unrelated_rerun():
    service = DashboardServiceStub()
    app = _start(service)
    assert not service.live_calls
    _button(app, "Consultar fuente oficial NHTSA").click().run()
    assert any("Selecciona marca" in item.value for item in app.error)
    assert not service.live_calls
    _select_official(app)
    _button(app, "Consultar fuente oficial NHTSA").click().run()
    assert service.live_calls == [("ford", "explorer", 2017)]
    assert any("1 recall(s)" in item.value for item in app.success)
    assert any("2026-09-18T00:00:00Z" in item.value for item in app.caption)
    assert any("26V000001" in item.label for item in app.expander)
    app.selectbox(key="ar_marca").select("ford").run()
    assert len(service.live_calls) == 1
    assert any("1 recall(s)" in item.value for item in app.success)
    app.selectbox(key="ar_real_make").select("chevrolet").run()
    assert app.selectbox(key="ar_real_model").value == PLACEHOLDER
    assert app.selectbox(key="ar_real_year").value == PLACEHOLDER
    assert "ar_real_response" not in app.session_state
    assert not app.exception


def test_official_recalls_network_error_clears_previous_response():
    service = DashboardServiceStub()
    app = _start(service)
    _select_official(app)
    _button(app, "Consultar fuente oficial NHTSA").click().run()
    assert "ar_real_response" in app.session_state
    service.live_error = True
    _button(app, "Consultar fuente oficial NHTSA").click().run()
    assert any("fuente oficial no pudo responder" in item.value for item in app.error)
    assert "ar_real_response" not in app.session_state
    assert not app.exception


def test_zero_official_recalls_are_not_presented_as_safety_guarantee():
    app = _start(DashboardServiceStub(live_count=0))
    _select_official(app)
    _button(app, "Consultar fuente oficial NHTSA").click().run()
    assert any("no demuestra ausencia de riesgos" in item.value for item in app.info)
    assert "ar_current_result" not in app.session_state
