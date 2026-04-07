#!/usr/bin/env python3
"""Tests for ``PolarCorrectionMLP`` and training helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from foilform.polar_correction_mlp import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    GEOM_STATIONS,
    N_SLOTS,
    POLAR_DIM,
    PolarCorrectionMLP,
    split_cl_cd,
)


def _load_train_script():
    path = _REPO / "scripts" / "train_polar_correction_mlp.py"
    spec = importlib.util.spec_from_file_location("train_polar_correction_mlp", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPolarCorrectionMLP(unittest.TestCase):
    def test_expected_parameter_count(self) -> None:
        m = PolarCorrectionMLP()
        self.assertEqual(m.count_parameters(), EXPECTED_PARAMETER_COUNT)
        self.assertEqual(EXPECTED_PARAMETER_COUNT, 34290)

    def test_forward_shape(self) -> None:
        m = PolarCorrectionMLP()
        g = torch.randn(4, GEOM_STATIONS, 2)
        p = torch.randn(4, POLAR_DIM)
        delta = m(g, p)
        self.assertEqual(tuple(delta.shape), (4, POLAR_DIM))

    def test_predict_is_base_plus_delta(self) -> None:
        m = PolarCorrectionMLP()
        g = torch.randn(3, GEOM_STATIONS, 2)
        b = torch.randn(3, POLAR_DIM)
        d = m(g, b)
        self.assertTrue(torch.allclose(m.predict(g, b), b + d))

    def test_initial_delta_is_near_zero(self) -> None:
        m = PolarCorrectionMLP()
        g = torch.randn(8, GEOM_STATIONS, 2) * 0.01
        b = torch.randn(8, POLAR_DIM) * 0.01
        d = m(g, b)
        self.assertLess(float(d.abs().max()), 0.1)

    def test_split_cl_cd(self) -> None:
        y = torch.arange(34, dtype=torch.float32).reshape(1, 34)
        cl, cd = split_cl_cd(y)
        self.assertEqual(cl.shape, (1, N_SLOTS))
        self.assertEqual(cd.shape, (1, N_SLOTS))
        self.assertEqual(int(cl[0, 0].item()), 0)
        self.assertEqual(int(cd[0, 0].item()), 17)


class TestTrainScriptHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tr = _load_train_script()

    def test_build_geom_xy_shape(self) -> None:
        tr = self.tr
        coords = np.random.randn(2, GEOM_STATIONS, 2).astype(np.float32)
        xy = tr.build_geom_xy(coords)
        self.assertEqual(xy.shape, (2, GEOM_STATIONS, 2))
        np.testing.assert_allclose(xy, coords)

    def test_build_geom_xy_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            self.tr.build_geom_xy(np.zeros((1, 100, 2), dtype=np.float32))

    def test_build_polar34_and_mask(self) -> None:
        tr = self.tr
        pol = np.full((2, 3, 3), np.nan, dtype=np.float32)
        pol[0, 0, :] = [0.0, 0.5, 0.02]
        pol[0, 1, :] = [5.0, 1.0, 0.03]
        pol[1, 2, :] = [10.0, 1.5, 0.04]
        y, m = tr.build_polar34_and_mask(pol)
        self.assertEqual(y.shape, (2, POLAR_DIM))
        self.assertAlmostEqual(float(y[0, 0]), 0.5)
        self.assertAlmostEqual(float(y[0, N_SLOTS]), 0.02)
        self.assertEqual(float(m[0, 0]), 1.0)
        self.assertEqual(float(m[1, 0]), 0.0)
        self.assertEqual(float(m[1, 2]), 1.0)

    def test_build_polar34_too_many_columns(self) -> None:
        pol = np.zeros((1, N_SLOTS + 1, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.tr.build_polar34_and_mask(pol)

    def test_masked_mse(self) -> None:
        tr = self.tr
        pred = torch.zeros(2, POLAR_DIM)
        tgt = torch.ones(2, POLAR_DIM)
        mask = torch.zeros(2, POLAR_DIM)
        mask[0, 0] = 1.0
        mask[0, N_SLOTS] = 1.0
        loss = tr.masked_mse(pred, tgt, mask)
        self.assertAlmostEqual(float(loss.item()), 1.0)

    def test_airfoil_train_val_mask(self) -> None:
        train, val = self.tr.airfoil_train_val_mask(10, 0.6, seed=0)
        self.assertEqual(int(train.sum() + val.sum()), 10)
        self.assertEqual(int(train.sum()), 6)

    def test_polar_corr_dataset(self) -> None:
        tr = self.tr
        n = 3
        geom = np.random.randn(n, GEOM_STATIONS, 2).astype(np.float32)
        tgt = np.random.randn(n, POLAR_DIM).astype(np.float32)
        m = np.ones((n, POLAR_DIM), dtype=np.float32)
        base = np.random.randn(n, POLAR_DIM).astype(np.float32)
        ds = tr.PolarCorrDataset(geom, tgt, m, base)
        self.assertEqual(len(ds), n)
        g, t, mk, b = ds[0]
        self.assertEqual(tuple(g.shape), (GEOM_STATIONS, 2))
        self.assertEqual(tuple(b.shape), (POLAR_DIM,))

    def test_evaluate_runs(self) -> None:
        tr = self.tr
        n = 4
        geom = np.random.randn(n, GEOM_STATIONS, 2).astype(np.float32)
        tgt = np.random.randn(n, POLAR_DIM).astype(np.float32)
        m = np.ones((n, POLAR_DIM), dtype=np.float32)
        base = np.random.randn(n, POLAR_DIM).astype(np.float32)
        ds = tr.PolarCorrDataset(geom, tgt, m, base)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model = PolarCorrectionMLP()
        mse, mae_cl, mae_cd = tr.evaluate(model, loader, torch.device("cpu"))
        self.assertTrue(np.isfinite(mse))
        self.assertTrue(np.isfinite(mae_cl))
        self.assertTrue(np.isfinite(mae_cd))

    def test_column_indices_for_airfoils(self) -> None:
        tr = self.tr
        pol = np.full((2, 5, 3), np.nan, dtype=np.float32)
        pol[0, 0, 1] = 1.0
        pol[0, 2, 1] = 2.0
        pol[1, 4, 1] = 3.0
        indices = tr.column_indices_for_airfoils(pol)
        self.assertEqual(indices[0], [0, 2])
        self.assertEqual(indices[1], [4])


if __name__ == "__main__":
    unittest.main()
