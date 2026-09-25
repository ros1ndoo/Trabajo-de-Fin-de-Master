"""El inventario independiente no debe fabricar potencia, identidades ni etiquetas."""

import io
import zipfile

import pandas as pd
import pytest

from auto_reliability.config import ProjectPaths
from auto_reliability.data_sources import DataSourceError
from auto_reliability.fuel_economy import acquire_inventory, read_inventory

VEHICLE = '<vehicle><id>1</id><make>Audi</make><model>A3</model><year>2025</year><cylinders>4</cylinders><VClass>Compact Cars</VClass></vehicle>'


def archive_bytes(xml: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('vehicles.xml', xml)
    return buffer.getvalue()


def test_recent_inventory_is_not_automatically_prediction_eligible(tmp_path):
    archive = tmp_path / 'source.zip'
    archive.write_bytes(archive_bytes('<vehicles>' + VEHICLE + '</vehicles>'))
    frame = read_inventory(archive)
    assert frame.year.tolist() == [2025]
    assert pd.isna(frame.power_cv.iloc[0])
    assert frame.cylinders.tolist() == [4]
    assert not frame.prediction_eligible.any()
    assert frame.recall_status.tolist() == ['not_queried']
    assert frame.identity_status.tolist() == ['pending_nhtsa_resolution']


@pytest.mark.parametrize('xml', [
    '<vehicles>' + VEHICLE + VEHICLE + '</vehicles>',
    '<vehicles>' + VEHICLE.replace('2025', '2025.5') + '</vehicles>',
    '<vehicles>' + VEHICLE.replace('<make>Audi</make>', '') + '</vehicles>',
    '<vehicles/>', '<vehicles>',
    '<!DOCTYPE vehicles [<!ENTITY x "unsafe">]><vehicles/>',
])
def test_invalid_or_unsafe_source_is_rejected(tmp_path, xml):
    archive = tmp_path / 'source.zip'
    archive.write_bytes(archive_bytes(xml))
    with pytest.raises(DataSourceError):
        read_inventory(archive)


@pytest.mark.parametrize('value,status', [('', 'missing'), ('bad', 'invalid'), ('-1', 'invalid'), ('0', 'observed')])
def test_missing_cylinders_are_not_assumed_to_be_zero(tmp_path, value, status):
    archive = tmp_path / 'source.zip'
    archive.write_bytes(archive_bytes('<vehicles>' + VEHICLE.replace('<cylinders>4', '<cylinders>' + value) + '</vehicles>'))
    frame = read_inventory(archive)
    assert frame.cylinders_status.iloc[0] == status
    assert pd.isna(frame.cylinders.iloc[0]) if status != 'observed' else frame.cylinders.iloc[0] == 0


def test_download_reuses_snapshot_and_detects_tampering(tmp_path, monkeypatch):
    payload = archive_bytes('<vehicles>' + VEHICLE + '</vehicles>')

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield payload

    monkeypatch.setattr('auto_reliability.fuel_economy.requests.get', lambda *a, **k: Response())
    paths = ProjectPaths(tmp_path)
    first = acquire_inventory(paths)

    def unexpected_network(*args, **kwargs):
        raise AssertionError('Cached acquisition must not use the network')

    monkeypatch.setattr('auto_reliability.fuel_economy.requests.get', unexpected_network)
    assert acquire_inventory(paths) == first
    assert not paths.gold_path.exists()
    source = next(paths.raw_dir.rglob('vehicles.xml.zip'))
    source.write_bytes(b'tampered fixture')
    with pytest.raises(DataSourceError, match='hash mismatch'):
        acquire_inventory(paths)


def test_xml_size_limit_and_multiple_members(tmp_path, monkeypatch):
    archive = tmp_path / 'source.zip'
    archive.write_bytes(archive_bytes('<vehicles>' + VEHICLE + '</vehicles>'))
    monkeypatch.setattr('auto_reliability.fuel_economy.MAX_XML_BYTES', 1)
    with pytest.raises(DataSourceError, match='size limit'):
        read_inventory(archive)
    with zipfile.ZipFile(archive, 'a') as stream:
        stream.writestr('second.xml', '<vehicles/>')
    with pytest.raises(DataSourceError, match='one XML'):
        read_inventory(archive)


def test_invalid_download_does_not_publish_raw_snapshot(tmp_path, monkeypatch):
    class BadResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b'<html>Provider error</html>'

    monkeypatch.setattr('auto_reliability.fuel_economy.requests.get', lambda *a, **k: BadResponse())
    paths = ProjectPaths(tmp_path)
    with pytest.raises(DataSourceError):
        acquire_inventory(paths)
    assert not list(paths.raw_dir.rglob('vehicles.xml.zip'))
    assert not list(paths.raw_dir.rglob('manifest.json'))
