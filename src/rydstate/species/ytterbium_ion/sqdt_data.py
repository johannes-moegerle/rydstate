from pathlib import Path

from rydstate.qdt.sqdt import ModelSetSQDT, register_model_set_sqdt

DIRECTORY = Path(__file__).parent

# Yb II (Yb+) energy levels, referenced to the Yb+ ground state (4f14.6s 2S1/2)
NIST_DATA_FILE = DIRECTORY / "nist_data.txt"

model_set_sqdt_ytterbium171_ion = ModelSetSQDT("Yb171_ion", nist_data_file=NIST_DATA_FILE)
register_model_set_sqdt(model_set_sqdt_ytterbium171_ion)

model_set_sqdt_ytterbium173_ion = ModelSetSQDT("Yb173_ion", nist_data_file=NIST_DATA_FILE)
register_model_set_sqdt(model_set_sqdt_ytterbium173_ion)

model_set_sqdt_ytterbium174_ion = ModelSetSQDT("Yb174_ion", nist_data_file=NIST_DATA_FILE)
register_model_set_sqdt(model_set_sqdt_ytterbium174_ion)
