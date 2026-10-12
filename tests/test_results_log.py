import csv
from pathlib import Path

import pytest

from gaze_cursor.eval.results_log import COLUMNS, append_rows

OLD_HEADER = ("time,session,model,features_set,alpha,n_train,n_test,err_px_mean,err_px_median,"
              "err_cm_mean,err_deg_mean,err_deg_median,jitter_px,jitter_deg")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_migrates_old_file_keeping_rows_and_defaulting_mode(tmp_path: Path) -> None:
    p = tmp_path / "results.csv"
    p.write_text(OLD_HEADER + "\nt0,s1,ridge,iris,1.0,400,500,300.0,280.0,5.9,5.5,5.2,120.0,2.2\n")
    append_rows([{"time": "t1", "mode": "losos", "confirmatory": 1, "config": "gaze_a10",
                  "err_deg_mean": 4.5, "ci_lo": 4.0, "ci_hi": 5.0}], p)
    assert tuple(csv.reader(p.open()).__next__()) == COLUMNS
    old, new = _rows(p)
    assert old["session"] == "s1" and old["err_deg_mean"] == "5.5" and old["mode"] == "within"
    assert old["confirmatory"] == "0" and old["ci_lo"] == ""
    assert new["mode"] == "losos" and new["confirmatory"] == "1" and new["err_deg_mean"] == "4.5000"


def test_appends_without_rewriting_when_header_matches(tmp_path: Path) -> None:
    p = tmp_path / "results.csv"
    append_rows([{"time": "a", "mode": "within"}], p)
    append_rows([{"time": "b", "mode": "loto"}], p)
    assert [r["time"] for r in _rows(p)] == ["a", "b"]


def test_unknown_column_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        append_rows([{"time": "a", "not_a_column": 1}], tmp_path / "r.csv")
