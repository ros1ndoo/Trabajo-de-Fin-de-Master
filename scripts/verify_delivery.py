"""Comprueba la entrega congelada sin red, entrenamiento ni cambios en los datos."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader

from auto_reliability.config import ProjectPaths
from auto_reliability.dashboard import summary_frame
from auto_reliability.releases import FILES, serving_paths
from auto_reliability.reporting import build_prediction_report_pdf
from auto_reliability.service import ReliabilityService, VehicleNotFoundError
from auto_reliability.storage import atomic_json


def verify(root: Path, output: Path) -> dict[str, Any]:
    """Valida hashes, métricas conservadas, inferencia real y exportaciones locales."""
    paths = ProjectPaths(root.resolve())
    frozen = json.loads((paths.root / 'artifacts/delivery_freeze.json').read_text(encoding='utf-8'))
    active = json.loads((paths.root / 'releases/active.json').read_text(encoding='utf-8'))
    if active['release'] != frozen['release']:
        raise ValueError('La publicación activa difiere de la versión congelada.')
    for name, expected in frozen['files'].items():
        if hashlib.sha256((paths.root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Archivo congelado modificado: {name}')
    bundle = serving_paths(paths)
    for name in FILES:
        if (paths.root / name).read_bytes() != (bundle.root / name).read_bytes():
            raise ValueError(f'La copia de trabajo difiere de la publicación: {name}')
    metrics = json.loads((bundle.artifacts_dir / 'model_metrics.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(paths.artifacts_dir / 'test_predictions.csv')
    errors = predictions.indice_fiabilidad_100 - predictions.prediccion_indice_100
    calculated = {'mae': float(errors.abs().mean()), 'rmse': float(np.sqrt((errors ** 2).mean()))}
    for key, value in calculated.items():
        if abs(value - metrics['final_test'][key]) > .0002:
            raise ValueError(f'Métrica no reproducida desde las predicciones conservadas: {key}')
    if len(predictions) != metrics['final_test']['n']:
        raise ValueError('Denominador de test incoherente.')
    service = ReliabilityService(paths, auto_bootstrap_demo=False)
    status = service.status()
    if status.demo_mode or not status.model_available or status.gold_rows != 1359 or status.catalog_rows != 2213:
        raise ValueError('No se ha cargado la población real congelada.')
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for make, model, year in [('Toyota', 'Camry', 2017), ('Chevrolet', 'Malibu', 2017), ('Ford', 'Transit Wagon', 2017)]:
        result = service.predict(make, model, year)
        if result.get('fuente_demo') or not result.get('es_baseline'):
            raise ValueError('La inferencia no utiliza el baseline real.')
        csv_path = output / f'{make.lower()}-{year}.csv'
        summary_frame(result).to_csv(csv_path, index=False, encoding='utf-8-sig')
        recovered = pd.read_csv(csv_path)
        if abs(float(recovered.prediccion_indice_100.iloc[0]) - result['prediccion_indice_100']) > 1e-8:
            raise ValueError('La exportación CSV altera la puntuación.')
        payload = build_prediction_report_pdf(result, output / f'{make.lower()}-{year}.pdf')
        pdf = PdfReader(BytesIO(payload))
        text = '\n'.join(page.extract_text() for page in pdf.pages)
        if str(result['version_modelo']) not in text or 'proxy' not in text.lower():
            raise ValueError('El PDF omite trazabilidad o advertencias.')
        examples.append({'marca': make, 'modelo': model, 'ano': year,
                         'indice': result['prediccion_indice_100'], 'paginas_pdf': len(pdf.pages)})
    try:
        service.predict('NO_EXISTE', 'NO_EXISTE', 2017)
    except VehicleNotFoundError:
        pass
    else:
        raise ValueError('Un vehículo inexistente recibió una puntuación.')
    report = {'release': active['release'], 'gold': status.gold_rows, 'catalogo': status.catalog_rows,
              'metricas_recalculadas': calculated, 'n_test': len(predictions), 'ejemplos': examples,
              'red_utilizada': False, 'entrenamiento_realizado': False,
              'nota': 'Validación numérica y estructural; la revisión visual del PDF es independiente.'}
    atomic_json(output / 'verification.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ProjectPaths.discover().root)
    parser.add_argument('--output', type=Path, default=Path('output/validacion-final'))
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.root, arguments.output), ensure_ascii=False, indent=2))
