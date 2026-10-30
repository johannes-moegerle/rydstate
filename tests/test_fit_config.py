from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
from rydstate import units
from rydstate.fit import SQDTLevel, check_fit, fit_from_config, load_measurements, write_fit
from rydstate.fit.fit_config import _summary, main
from rydstate.qdt import ModelSQDT

FITS_DIRECTORY = Path(__file__).parents[1] / "fits"

TRUE_NP = ModelSQDT("Cs", "np_3/2", [3.5590676, 0.37469])
TRUE_NS = ModelSQDT("Cs", "ns_1/2", [4.0493532, 0.2391])
SIGMA_MHZ = 1e-3


def mhz(value_au: float) -> float:
    return units.au_to_user(value_au, "energy", "MHz")


def write_dataset(directory: Path, rows: list[dict[str, str]], reference: str = "Test 2026, J. Test 1, 1") -> None:
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({"species": "Cs", "reference": reference}))
    with (directory / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["level", "lower_level", "value", "uncertainty", "unit", "source_label"]
        )
        writer.writeheader()
        writer.writerows(rows)


def level_rows(model: ModelSQDT, ns: range, shift_mhz: float = 0) -> list[dict[str, str]]:
    return [
        {
            "level": f"{n}p_3/2",
            "lower_level": "",
            "value": repr(mhz(model.calc_energy_au(n=n)) + shift_mhz),
            "uncertainty": repr(SIGMA_MHZ),
            "unit": "MHz",
            "source_label": f"{n}P3/2",
        }
        for n in ns
    ]


def interval_rows(ns: range) -> list[dict[str, str]]:
    return [
        {
            "level": f"{n + 4}s_1/2",
            "lower_level": f"{n}p_3/2",
            "value": repr(mhz(TRUE_NS.calc_energy_au(n=n + 4) - TRUE_NP.calc_energy_au(n=n))),
            "uncertainty": repr(SIGMA_MHZ),
            "unit": "MHz",
            "source_label": f"{n}P3/2 -> {n + 4}S1/2",
        }
        for n in ns
    ]


def write_config(path: Path, datasets: dict[str, dict[str, object]], **kwargs: object) -> Path:
    config = {
        "species": "Cs",
        "data_directory": "data",
        "models_directory": "models",
        "report_file": "results/test.json",
        "series": {"np_3/2": {"orders": 2, "n_min": 19}, "ns_1/2": {"orders": 2, "n_min": 20}},
        "datasets": datasets,
        **kwargs,
    }
    path.write_text(json.dumps(config))
    return path


def test_level_from_label() -> None:
    assert SQDTLevel.from_label("45p_3/2", "Cs") == SQDTLevel("np_3/2", 45)
    assert SQDTLevel.from_label("5s 40d 1D2", "Sr88") == SQDTLevel("5s nd 1D2", 40)
    level = SQDTLevel.from_label("15h_11/2", "Cs")  # the channel is stored with its canonical label
    assert level == SQDTLevel("nh_5.5", 15)
    assert SQDTLevel.from_label(str(level), "Cs") == level
    with pytest.raises(ValueError, match="no short label of a level"):
        SQDTLevel.from_label("np_3/2", "Cs")


def test_load_measurements(tmp_path: Path) -> None:
    write_dataset(tmp_path / "test2026", level_rows(TRUE_NP, range(30, 33)) + interval_rows(range(40, 42)))
    measurements = load_measurements(tmp_path / "test2026")

    assert len(measurements) == 5
    first, last = measurements[0], measurements[-1]
    assert first.upper == SQDTLevel("np_3/2", 30)
    assert first.lower is None
    assert first.value_au == pytest.approx(TRUE_NP.calc_energy_au(n=30), rel=1e-15)
    assert first.uncertainty_au == pytest.approx(units.user_to_au(SIGMA_MHZ, "MHz", "energy"))
    assert first.dataset == "test2026"
    assert first.label == "30P3/2"
    assert first.reference == "Test 2026, J. Test 1, 1"
    assert last.lower == SQDTLevel("np_3/2", 41)

    rows = level_rows(TRUE_NP, range(30, 32))
    rows[1]["source_label"] = rows[0]["source_label"]
    write_dataset(tmp_path / "duplicate", rows)
    with pytest.raises(ValueError, match="not unique"):
        load_measurements(tmp_path / "duplicate")

    for unit in ("nm", "degC"):  # a wavelength, or a unit that is not an energy at all
        rows = level_rows(TRUE_NP, range(30, 32))
        rows[1]["unit"] = unit
        write_dataset(tmp_path / f"unit_{unit}", rows)
        with pytest.raises(ValueError, match=f"the unit '{unit}' of '31P3/2' is not an energy"):
            load_measurements(tmp_path / f"unit_{unit}")


def test_fit_from_config(tmp_path: Path) -> None:
    shift_mhz = 0.3
    rows = level_rows(TRUE_NP, range(30, 80, 3), shift_mhz=shift_mhz)
    rows[0]["value"] = "0.0"  # a wrong value, which is excluded
    write_dataset(tmp_path / "data" / "levels", rows, reference="Levels 2026, J. Test 1, 1")
    write_dataset(tmp_path / "data" / "intervals", interval_rows(range(40, 90, 5)), reference="Intervals 2026")
    config = write_config(
        tmp_path / "test.json",
        {
            "levels": {
                "free_offset": True,
                "exclude": {"30P3/2": "wrong value"},
                "systematic": {"value": 1, "unit": "kHz"},
                "comment": "test",
            },
            "intervals": {"n_min": 50},
        },
        series={"np_3/2": {"orders": 2, "n_min": 19, "start": [3.5]}, "ns_1/2": {"orders": 2, "n_min": 20}},
    )
    fit = fit_from_config(config)
    fitted_np, fitted_ns = fit.result.models

    assert fit.result.chi2 == pytest.approx(0, abs=1e-6)
    assert mhz(fit.result.offsets_au["levels"]) == pytest.approx(-shift_mhz, abs=1e-6)
    for fitted, true in ((fitted_np, TRUE_NP), (fitted_ns, TRUE_NS)):
        assert fitted.quantum_defects == pytest.approx(true.quantum_defects, rel=1e-7)
    assert fitted_np.n_range == (19, math.inf)
    assert fitted_np.reference == ("Levels 2026, J. Test 1, 1", "Intervals 2026")
    assert fitted_ns.reference == ("Intervals 2026",)
    assert fitted_np.fit_info is not None
    assert fitted_np.fit_info["config"] == f"{config.parent.parent.name}/{config.parent.name}/test.json"
    assert fit.datasets["levels"] == {"loaded": 17, "used": 16, "excluded": 1}
    assert fit.datasets["intervals"] == {"loaded": 10, "used": 8, "level below n_min": 2}
    # the systematic is added in quadrature
    used = [m for m in fit.result.measurements if m.dataset == "levels"]
    assert mhz(used[0].uncertainty_au) == pytest.approx(math.hypot(SIGMA_MHZ, 1e-3))

    assert check_fit(fit) != []  # nothing written yet
    files = write_fit(fit)
    assert {file.name for file in files} == {"np_3_2.json", "ns_1_2.json", "test.json"}
    assert check_fit(fit_from_config(config)) == []
    loaded = ModelSQDT.from_dict(json.loads((tmp_path / "models" / "np_3_2.json").read_text()))
    assert loaded.quantum_defects == fitted_np.quantum_defects
    report = json.loads((tmp_path / "results" / "test.json").read_text())
    assert report["dof"] == fit.result.dof
    assert "0 values with abs(residual) > 3 sigma" in _summary(fit)

    main([str(config), "--check"])
    (tmp_path / "models" / "np_3_2.json").write_text(json.dumps(TRUE_NS.to_dict() | {"channel": "np_3/2"}))
    with pytest.raises(SystemExit):
        main([str(config), "--check"])


def test_series_n_min(tmp_path: Path) -> None:
    """The models are fitted with the n range of the written model; levels below n_min are not used."""
    write_dataset(tmp_path / "data" / "levels", level_rows(TRUE_NP, range(30, 80, 3)))
    series = {"np_3/2": {"orders": 2, "n_min": 40}}
    fit = fit_from_config(write_config(tmp_path / "test.json", {"levels": {}}, series=series))
    assert fit.datasets["levels"] == {"loaded": 17, "used": 13, "level below n_min of its series": 4}
    assert fit.result.models[0].n_range == (40, math.inf)
    assert fit.result.models[0].quantum_defects == pytest.approx(TRUE_NP.quantum_defects, rel=1e-7)


def test_summary_lists_large_residuals(tmp_path: Path) -> None:
    rows = level_rows(TRUE_NP, range(30, 80, 3))
    rows[5]["value"] = repr(float(rows[5]["value"]) + 100 * SIGMA_MHZ)  # a value far off the series
    write_dataset(tmp_path / "data" / "levels", rows)
    summary = _summary(
        fit_from_config(
            write_config(tmp_path / "test.json", {"levels": {}}, series={"np_3/2": {"orders": 2, "n_min": 19}})
        )
    )
    outliers = summary.split("sigma (residual = fit - measured):\n")[1].splitlines()
    assert outliers[0].split()[:2] == ["levels", rows[5]["source_label"]]
    assert float(outliers[0].split()[2]) < -50


def test_config_errors(tmp_path: Path) -> None:
    write_dataset(tmp_path / "data" / "levels", level_rows(TRUE_NP, range(30, 40)))
    with pytest.raises(ValueError, match="unknown keys"):
        fit_from_config(write_config(tmp_path / "a.json", {"levels": {"free_ofset": True}}))
    config_file = write_config(tmp_path / "e.json", {"levels": {}})
    config_file.write_text(
        json.dumps({k: v for k, v in json.loads(config_file.read_text()).items() if k != "models_directory"})
    )
    with pytest.raises(ValueError, match=r"missing keys \['models_directory'\]"):
        fit_from_config(config_file)
    with pytest.raises(ValueError, match="not in the dataset"):
        fit_from_config(write_config(tmp_path / "b.json", {"levels": {"exclude": {"99P3/2": "?"}}}))
    with pytest.raises(ValueError, match="more start values than orders"):
        fit_from_config(
            write_config(
                tmp_path / "d.json", {"levels": {}}, series={"np_3/2": {"orders": 2, "n_min": 19, "start": [3, 0, 0]}}
            )
        )
    with pytest.raises(ValueError, match="use the label"):
        fit_from_config(
            write_config(tmp_path / "c.json", {"levels": {}}, series={"np_1.5": {"orders": 2, "n_min": 19}})
        )


@pytest.mark.parametrize("config", sorted(FITS_DIRECTORY.glob("*/fit_config.json")), ids=lambda path: path.parent.name)
def test_fits_reproduce_the_models(config: Path) -> None:
    """The fits of fits/ reproduce the models of the species (rerun them after changing the data or the config)."""
    assert check_fit(fit_from_config(config)) == []
