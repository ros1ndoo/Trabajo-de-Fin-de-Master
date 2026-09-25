"""Regresiones de cierre sobre la entrega real, sin red ni entrenamiento."""

import json
import runpy
from pathlib import Path

import pytest

from auto_reliability.dashboard import _inject_styles

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_delivery_reproduces_metrics_and_exports(tmp_path):
    verify = runpy.run_path(str(ROOT / 'scripts/verify_delivery.py'))['verify']
    report = verify(ROOT, tmp_path)
    assert report['n_test'] == 819
    assert report['gold'] == 1359
    assert report['metricas_recalculadas']['mae'] == pytest.approx(12.1537, abs=.0001)
    assert len(list(tmp_path.glob('*.pdf'))) == 3
    assert len(list(tmp_path.glob('*.csv'))) == 3


def test_changed_active_release_fails_before_loading_model(tmp_path):
    verify = runpy.run_path(str(ROOT / 'scripts/verify_delivery.py'))['verify']
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'releases').mkdir()
    (tmp_path / 'artifacts/delivery_freeze.json').write_text(json.dumps({'release': 'frozen'}))
    (tmp_path / 'releases/active.json').write_text(json.dumps({'release': 'changed'}))
    with pytest.raises(ValueError, match='publicación activa'):
        verify(tmp_path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_desktop_styles_preserve_mobile_breakpoints_and_keyboard_focus():
    class Capture:
        def markdown(self, text, **kwargs):
            self.styles = text

    capture = Capture()
    _inject_styles(capture)
    assert '@media (min-width: 951px)' in capture.styles
    assert '@media (max-width: 950px)' in capture.styles
    assert '@media (max-width: 700px)' in capture.styles
    assert ':focus-visible' in capture.styles
