"""Tests for the Aimless runner and the Pointless log parser (real log excerpts)."""

from __future__ import annotations

from pathlib import Path

from diffraction_lab.aimless_runner import aimless_keywords, parse_aimless_summary
from diffraction_lab.pointless_runner import is_identity_reindex, parse_pointless_log

# Aimless 0.8 summary for ProcessingLab thaumatin frames 1–100.
AIMLESS_SUMMARY = """
<!--SUMMARY_BEGIN--> $TEXT:Result: $$ $$
Summary data for        Project: UnspecifiedProject Crystal: UnspecifiedCrystal Dataset: UnspecifiedDataset

                                           Overall  InnerShell  OuterShell
Low resolution limit                       54.06     54.06      1.18
High resolution limit                       1.16      6.36      1.16

Rmerge  (within I+/I-)                     0.103     0.064     0.000
Rmeas (all I+ & I-)                        0.148     0.089     0.000
Rmerge in top intensity bin                0.057        -         -
Total number of observations               48017       604        24
Mean((I)/sd(I))                              3.5       9.2       0.8
Mn(I) half-set correlation CC(1/2)         0.982     0.962     0.000
Completeness                                30.8      47.5       0.5
Multiplicity                                 1.8       2.1       1.0

The anomalous signal appears to be weak so anomalous flag was left OFF

Estimates of resolution limits: overall
   from half-dataset correlation CC(1/2) >  0.30: limit =  1.45A
   from Mn(I/sd) >  1.50:                         limit =  1.54A

Estimates of resolution limits in reciprocal lattice directions:
  Along h k plane
   from half-dataset correlation CC(1/2) >  0.30: limit =  1.49A

Average unit cell:    57.94   57.94  150.24   90.00   90.00   90.00
Space group: P 4 2 2
Minimum and maximum SD correction factors: Fulls   0.69   6.61 Partials   0.00   0.00

$$ <!--SUMMARY_END-->
"""

POINTLESS_POINT_GROUP = """
Checking for possible twinning
L-test for twinning (acentrics only) to maximum resolution    1.963
      Estimated twin fraction alpha from cumulative N(|L|) plot 0.027 (+/-0.004)
   < |L| >:    0.480 (0.5 untwinned, 0.375 perfect twin)
      Estimated twin fraction alpha from < |L| >    0.024
   < L^2 >:    0.310 (0.333 untwinned, 0.2 perfect twin)
      Estimated twin fraction alpha from < L^2 >    0.022

The L-test suggests that the data are not twinned
Space group confidence (= Sqrt(Score * (Score - NextBestScore))) =     0.00
Laue group confidence  (= Sqrt(Score * (Score - NextBestScore))) =     1.00
<!--SUMMARY_BEGIN-->
Best Solution:    point group P 4 2 2

   Reindex operator:                   [h,k,l]
   Laue group probability:             1.000
   Space group confidence:             0.000
   Laue group confidence               0.999
"""

POINTLESS_SPACE_GROUP = """
Best Solution:    space group P 41 21 2

   Reindex operator:                   [k,h,-l]
   Space group confidence:             0.931
   Laue group confidence               0.985
"""


def test_parse_aimless_summary() -> None:
    s = parse_aimless_summary(AIMLESS_SUMMARY)
    assert s.high_resolution == 1.16
    assert s.value("High resolution limit", "inner") == 6.36
    assert s.rmeas == 0.148 and s.cc_half == 0.982
    assert s.mean_i_over_sigma == 3.5 and s.completeness == 30.8 and s.multiplicity == 1.8
    # "-" entries are missing values.
    assert s.value("Rmerge in top intensity bin") == 0.057
    assert s.value("Rmerge in top intensity bin", "outer") is None
    # Only the overall estimates; the per-direction ones are not mixed in.
    assert s.resolution_estimates == {"half-dataset correlation CC(1/2) >  0.30": 1.45,
                                      "Mn(I/sd) >  1.50": 1.54}
    assert s.space_group == "P 4 2 2"
    assert s.cell == (57.94, 57.94, 150.24, 90.0, 90.0, 90.0)
    assert "Minimum and maximum SD correction factors: Fulls" not in " ".join(s.rows)


def test_parse_aimless_summary_missing() -> None:
    s = parse_aimless_summary("AIMLESS: *** error: no reflections\n")
    assert not s.rows and s.high_resolution is None


def test_aimless_keywords() -> None:
    assert aimless_keywords() == ["! DiffractionLab Aimless (defaults)"]
    assert aimless_keywords(resolution_high=1.45, anomalous=True, custom="CYCLES 10\n") == [
        "RESOLUTION HIGH 1.45", "ANOMALOUS ON", "CYCLES 10"]


def test_parse_pointless_point_group(tmp_path: Path) -> None:
    log = tmp_path / "pointless.log"
    log.write_text(POINTLESS_POINT_GROUP)
    r = parse_pointless_log(log)
    assert r["space_group"] == "P 4 2 2" and r["solution_kind"] == "point group"
    assert r["reindex_operator"] == "[h,k,l]"
    assert r["confidence"] == 0.0 and r["laue_confidence"] == 0.999
    twin = r["twinning"]
    assert (twin.mean_abs_l, twin.mean_l2, twin.twin_fraction) == (0.48, 0.31, 0.024)
    assert twin.verdict == "The L-test suggests that the data are not twinned"
    assert twin.suggests_twinning is False


def test_parse_pointless_space_group(tmp_path: Path) -> None:
    log = tmp_path / "pointless.log"
    log.write_text(POINTLESS_SPACE_GROUP)
    r = parse_pointless_log(log)
    assert r["space_group"] == "P 41 21 2" and r["solution_kind"] == "space group"
    assert r["confidence"] == 0.931 and r["laue_confidence"] == 0.985


def test_twinning_verdicts() -> None:
    from diffraction_lab.pointless_runner import TwinningTest

    assert TwinningTest(0.42, 0.25, 0.2, "The L-test suggests that the data may be twinned").suggests_twinning
    assert TwinningTest(None, None, None, "The data do not appear to be twinned, from the L-test"
                        ).suggests_twinning is False
    assert TwinningTest(None, None, None, None).suggests_twinning is None


def test_is_identity_reindex() -> None:
    assert is_identity_reindex("[h,k,l]") and is_identity_reindex(" [H, K, L] ")
    assert not is_identity_reindex("[k,h,-l]")
    assert not is_identity_reindex(None)
