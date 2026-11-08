"""Pruebas unitarias para la logica de integracion y composicion gramatical de NAO."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.robot.app_inferencia_nao import componer_oracion_gramatical


def test_composicion_gramatical_tres_tokens() -> None:
    """Verifica la composicion fluida para tripletas sujeto-verbo-objeto."""
    oracion_1 = componer_oracion_gramatical(["YO", "NECESITAR", "AGUA"])
    assert oracion_1 == "Yo necesito agua."

    oracion_2 = componer_oracion_gramatical(["YO", "QUERER", "MEDICINA"])
    assert oracion_2 == "Yo quiero mi medicina."

    oracion_3 = componer_oracion_gramatical(["TU", "NECESITAR", "AYUDA"])
    assert oracion_3 == "Tu necesitas ayuda."


def test_composicion_gramatical_tokens_individuales() -> None:
    """Verifica la respuesta ante senas unicas de emergencia o necesidad."""
    assert componer_oracion_gramatical(["AYUDA"]) == "Por favor, necesito ayuda inmediata."
    assert componer_oracion_gramatical(["AGUA"]) == "Por favor, solicito un vaso de agua."
    assert componer_oracion_gramatical(["MEDICINA"]) == "Por favor, necesito tomar mi medicina."


def test_composicion_gramatical_bufer_vacio() -> None:
    """Verifica que un bufer vacio retorne una cadena vacia."""
    assert componer_oracion_gramatical([]) == ""
