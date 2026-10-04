"""Every metric key used by the new templates must exist in GLOSSARY."""
import unittest

from glossary import GLOSSARY

REQUIRED = {"surprise_pct", "pead_drift", "event_car", "event_vol_flag",
            "peer_percentile", "rv_quadrant", "short_pct_float", "days_to_cover",
            "si_mom_change", "squeeze_composite"}


class TestRadarGlossary(unittest.TestCase):
    def test_all_radar_keys_defined(self):
        missing = REQUIRED - set(GLOSSARY)
        self.assertEqual(missing, set())

    def test_entries_have_required_fields(self):
        for key in REQUIRED:
            entry = GLOSSARY[key]
            self.assertTrue(entry["term"])
            self.assertTrue(entry["short"])
            self.assertTrue(entry["long"])
