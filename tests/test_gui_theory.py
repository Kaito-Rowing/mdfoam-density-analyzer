from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import QApplication

from mdfoam_analyzer.gui import MainWindow
from mdfoam_analyzer.theory import evaporation_flux


class GuiTheorySettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_xlsx_preset_is_default_and_preserves_small_molecule_mass(self) -> None:
        window = MainWindow()
        try:
            settings = window.theory_settings()

            self.assertAlmostEqual(settings.rho_l, 956.2)
            self.assertAlmostEqual(settings.rho_v, 0.9409227266221003)
            self.assertAlmostEqual(settings.molecule_mass, 2.9915e-26)
            self.assertAlmostEqual(evaporation_flux(settings, 1.0), 139.676, delta=1.0e-3)
            self.assertIn("139.676", window.theory_diagnostics_label.text())
        finally:
            window.close()

    def test_reference_water_preset_updates_material_values(self) -> None:
        window = MainWindow()
        try:
            window.theory_preset_combo.setCurrentText("水 300K 参考")
            settings = window.theory_settings()

            self.assertAlmostEqual(settings.rho_l, 1000.0)
            self.assertAlmostEqual(settings.rho_v, 0.0256)
            self.assertAlmostEqual(settings.molecule_mass, 2.9915e-26)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
