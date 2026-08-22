import unittest

import torch

from tmp import methods


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(6, 8)

    def model(self, time_inputs=1):
        return methods.ActivationModel(8, 8, 16, 2, time_inputs)

    def test_baseline_losses_backpropagate(self):
        cases = {
            "additive_simple": methods.ActivationModel(8, hidden=32, simple=True),
            "additive_capacity": self.model(0),
            "interpolation": self.model(1),
            "glp": self.model(1),
        }
        for name, model in cases.items():
            with self.subTest(name=name):
                value = methods.loss(name, model, self.z)
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_consistency_uses_ema_target(self):
        model = self.model(1)
        target = methods.make_ema(model)
        value = methods.loss("consistency", model, self.z, target_model=target)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertFalse(any(p.grad is not None for p in target.parameters()))

    def test_meanflow_jvp_backpropagates(self):
        model = self.model(2)
        value = methods.loss("meanflow", model, self.z)
        value.backward()
        self.assertTrue(torch.isfinite(value))

    def test_rectified_flow_uses_frozen_teacher_path(self):
        model, teacher = self.model(1), self.model(1).eval()
        value = methods.loss("rectified", model, self.z, teacher=teacher)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertFalse(any(p.grad is not None for p in teacher.parameters()))

    def test_rectified_flow_reuses_precomputed_pairs(self):
        model = self.model(1)
        value = methods.loss("rectified", model, None, pair=(self.z, torch.randn_like(self.z)))
        value.backward()
        self.assertTrue(torch.isfinite(value))

    def test_repairs_keep_activation_shape(self):
        for name, time_inputs in (("additive_capacity", 0), ("interpolation", 1),
                                  ("glp", 1), ("consistency", 1), ("meanflow", 2)):
            with self.subTest(name=name):
                repaired, path = methods.repair(name, self.model(time_inputs), self.z)
                self.assertEqual(repaired.shape, self.z.shape)
                self.assertGreaterEqual(len(path), 2)

    def test_local_geometry_reconstructs_tangent_points(self):
        bank = torch.randn(40, 8)
        geometry = methods.local_geometry(bank, self.z[:2], k=16, rank=4)
        tangent, normal = methods.split_local(self.z[:2], geometry["basis"])
        self.assertEqual(geometry["spectrum"].shape, (2, 8))
        self.assertTrue(torch.allclose(tangent + normal, self.z[:2], atol=1e-6))

    def test_geometry_repairs_return_bank_or_path(self):
        bank = torch.randn(40, 8)
        nearest = methods.nearest(bank, self.z[:2])
        segment = methods.segment_nearest(bank, self.z[:2], self.z[:2] + 0.5)
        geodesic, path = methods.local_geodesic(
            bank, self.z[:2], torch.ones(8), 0.5, steps=2, k=16, rank=4)
        self.assertEqual(nearest.shape, self.z[:2].shape)
        self.assertEqual(segment.shape, self.z[:2].shape)
        self.assertEqual(geodesic.shape, self.z[:2].shape)
        self.assertEqual(len(path), 3)


if __name__ == "__main__":
    unittest.main()
