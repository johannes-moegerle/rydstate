from pathlib import Path

from rydstate.qdt.sqdt import ModelSetSQDT, register_model_set_sqdt

DIRECTORY = Path(__file__).parent

model_set_sqdt_strontium88 = ModelSetSQDT(
    "Sr88",
    models=DIRECTORY / "sqdt_models",
)
register_model_set_sqdt(model_set_sqdt_strontium88)
