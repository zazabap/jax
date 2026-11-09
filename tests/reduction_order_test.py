# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from absl.testing import absltest
import jax
from jax import lax
from jax._src import reduction_order as ro
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


def _eqns_by_name(jaxpr, name: str):
  return [e for e in jaxpr.eqns if e.primitive.name == name]


class ReductionOrderTest(jtu.JaxTestCase):

  def test_normalize_order_and_intrinsic_validation(self):
    self.assertIsNone(ro.normalize_order(None))
    self.assertEqual(ro.normalize_order("unordered"), "unordered")
    self.assertEqual(ro.normalize_order("linear"), "linear")
    intr = ro.Intrinsic("tpu_xlane_128")
    self.assertEqual(ro.normalize_order(intr), intr)
    for bad in ("", "tree", 123, ["linear"]):
      with self.assertRaises(ValueError):
        ro.normalize_order(bad)
    for bad_name in ("", None, 42):
      with self.assertRaises(ValueError):
        ro.Intrinsic(bad_name)  # type: ignore[arg-type]
    for u in (ro.Unordered(), ro.Unordered(None), ro.Unordered("xla"), ro.Unordered("mosaic")):
      self.assertEqual(ro.normalize_order(u), u)
    self.assertEqual(ro.Unordered(), ro.Unordered(None))
    self.assertNotEqual(ro.Unordered("xla"), ro.Unordered("mosaic"))
    for bad_scope in ("", 123, ["xla"]):
      with self.assertRaises(ValueError):
        ro.Unordered(bad_scope)  # type: ignore[arg-type]

  def test_jaxpr_order_params_default_and_explicit(self):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 3), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    for fn, prim in [
        (lambda a: jnp.sum(a, axis=1), "reduce_sum"),
        (lambda a: jnp.max(a, axis=1), "reduce_max"),
        (lambda a: jnp.dot(a, w), "dot_general"),
        (lambda a: jnp.matmul(a, w), "dot_general"),
        (lambda a: jnp.einsum("ij,jk->ik", a, w), "dot_general"),
        (lambda a: lax.reduce_sum(a, (1,)), "reduce_sum"),
        (lambda a: lax.reduce_max(a, (1,)), "reduce_max"),
        (lambda a: lax.dot(a, w), "dot_general"),
        (lambda a: lax.dot_general(a, w, dnums), "dot_general"),
    ]:
      eqns = _eqns_by_name(jax.make_jaxpr(fn)(x), prim)
      self.assertLen(eqns, 1)
      self.assertNotIn("reduction_order", eqns[0].params)

    for ord_val in ("unordered", "linear", ro.Intrinsic("test_hw")):
      for fn, prim in [
          (lambda a: jnp.sum(a, axis=1, reduction_order=ord_val), "reduce_sum"),
          (lambda a: jnp.max(a, axis=1, reduction_order=ord_val), "reduce_max"),
          (lambda a: jnp.dot(a, w, reduction_order=ord_val), "dot_general"),
          (lambda a: jnp.matmul(a, w, reduction_order=ord_val), "dot_general"),
          (lambda a: lax.reduce_sum(a, (1,), reduction_order=ord_val), "reduce_sum"),
          (lambda a: lax.reduce_max(a, (1,), reduction_order=ord_val), "reduce_max"),
          (lambda a: lax.dot(a, w, reduction_order=ord_val), "dot_general"),
          (lambda a: lax.dot_general(a, w, dnums, reduction_order=ord_val), "dot_general"),
      ]:
        eqn, = _eqns_by_name(jax.make_jaxpr(fn)(x), prim)
        self.assertEqual(eqn.params["reduction_order"], ord_val)

  def test_linear_sequential_ieee754_exactness(self):
    # ((1 + 1e8) + 1) - 1e8 == 0.0 in f32 left-to-right, whereas reassociated is 2.0.
    v = jnp.array([1.0, 1e8, 1.0, -1e8], dtype=jnp.float32)
    expected = np.float32(0.0)
    for fn in (
        lambda a: jnp.sum(a, reduction_order="linear"),
        jax.jit(lambda a: jnp.sum(a, reduction_order="linear")),
        lambda a: jnp.dot(a, jnp.ones(4, dtype=jnp.float32), reduction_order="linear"),
        jax.jit(lambda a: jnp.einsum("i->", a, reduction_order="linear")),
    ):
      out = np.asarray(fn(v))
      self.assertEqual(out.view(np.uint32), expected.view(np.uint32))

    # Starting from x[0] preserves -0.0 on all -0.0 inputs.
    neg_zeros = jnp.array([-0.0, -0.0, -0.0], dtype=jnp.float32)
    nz_out = np.asarray(jax.jit(lambda a: jnp.sum(a, reduction_order="linear"))(neg_zeros))
    self.assertTrue(np.signbit(nz_out))

    # Empty K=0 reduction returns +0.0.
    empty = jnp.zeros((2, 0), dtype=jnp.float32)
    self.assertArraysEqual(
        jnp.sum(empty, axis=1, reduction_order="linear"),
        jnp.zeros((2,), dtype=jnp.float32))
    self.assertArraysEqual(
        jnp.dot(empty, jnp.zeros((0, 3), dtype=jnp.float32), reduction_order="linear"),
        jnp.zeros((2, 3), dtype=jnp.float32))

    # 0-ULP separate-rounding IEEE-754 fl(acc + fl(a * b)) without FMA contraction (<=8KiB & >8KiB).
    u = np.float32(1.0 + 2.0**-23)
    fma_a, fma_b = jnp.array([u, -u], jnp.float32), jnp.array([u, u], jnp.float32)
    self.assertEqual(float(jax.jit(lambda x, y: jnp.dot(x, y, reduction_order="linear"))(fma_a, fma_b)), 0.0)
    big_a = jnp.broadcast_to(fma_a, (64, 64, 2))
    big_b = jnp.broadcast_to(fma_b[:, None], (2, 2))
    self.assertArraysEqual(
        jax.jit(lambda x, y: jnp.matmul(x, y, reduction_order="linear"))(big_a, big_b),
        jnp.zeros((64, 64, 2), dtype=jnp.float32))

  def test_intrinsic_registration_and_unregistered_error(self):
    x = jnp.arange(8, dtype=jnp.float32)
    unreg = ro.Intrinsic("unregistered_test_leaf")
    jaxpr = jax.make_jaxpr(lambda a: jnp.sum(a, reduction_order=unreg))(x)
    self.assertEqual(_eqns_by_name(jaxpr, "reduce_sum")[0].params["reduction_order"], unreg)
    with self.assertRaises(NotImplementedError):
      jax.jit(lambda a: jnp.sum(a, reduction_order=unreg))(x)

    name = "test_registered_leaf"
    ro.register_intrinsic(
        name,
        lambda a, *, axes, op: (
            lax.reduce_sum(a, axes, reduction_order="linear") if op == "sum"
            else lax.reduce_max(a, axes, reduction_order="linear")))
    try:
      leaf = ro.Intrinsic(name)
      self.assertArraysEqual(
          jax.jit(lambda a: jnp.sum(a, reduction_order=leaf))(x),
          jnp.sum(x, reduction_order="linear"))
      self.assertArraysEqual(
          jax.jit(lambda a: jnp.max(a, reduction_order=leaf))(x),
          jnp.max(x, reduction_order="linear"))
    finally:
      ro._INTRINSIC_IMPLS.pop(name, None)

  def test_einsum_single_reduction_and_errors(self):
    a = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    b = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    c = jnp.ones((4, 2), dtype=jnp.float32)
    sq = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)

    j_sum = jax.make_jaxpr(lambda x: jnp.einsum("ij->i", x, reduction_order="linear"))(a)
    self.assertEqual(_eqns_by_name(j_sum, "reduce_sum")[0].params["reduction_order"], "linear")
    self.assertAllClose(jnp.einsum("ij->i", a, reduction_order="linear"), jnp.sum(a, axis=1))

    j_dot = jax.make_jaxpr(lambda x, y: jnp.einsum("ik,kj->ij", x, y, reduction_order="linear"))(a, b)
    self.assertEqual(_eqns_by_name(j_dot, "dot_general")[0].params["reduction_order"], "linear")
    self.assertEqual(_eqns_by_name(j_dot, "dot_general")[0].invars, list(j_dot.jaxpr.invars))
    self.assertAllClose(jnp.einsum("ik,kj->ij", a, b, reduction_order="linear"), a @ b)

    # Transposed-output einsum preserves user (arg0, arg1) operand order on dot_general.
    j_tr = jax.make_jaxpr(lambda x, y: jnp.einsum("ik,kj->ji", x, y, reduction_order="linear"))(a, b)
    self.assertEqual(_eqns_by_name(j_tr, "dot_general")[0].invars, list(j_tr.jaxpr.invars))
    self.assertAllClose(jnp.einsum("ik,kj->ji", a, b, reduction_order="linear"), (a @ b).T)

    # 0-reduction or repeated-index einsums must raise ValueError when reduction_order is not None.
    for fn in (
        lambda: jnp.einsum("ij->ji", a, reduction_order="linear"),
        lambda: jnp.einsum("ii->i", sq, reduction_order="linear"),
        lambda: jnp.einsum("ii->", sq, reduction_order="linear"),
        lambda: jnp.einsum("i,j->ij", a[0], b[0], reduction_order="linear"),
        lambda: jnp.einsum("ij,ij->ij", a, a, reduction_order="unordered"),
    ):
      with self.assertRaisesRegex(ValueError, "reduction_order="):
        fn()

    # >1-reduction einsums must raise ValueError when reduction_order is not None.
    for fn in (
        lambda: jnp.einsum("ab,bc,cd->ad", a, b, c, reduction_order="linear"),
        lambda: jnp.einsum("ij,jk->i", a, b, reduction_order="linear"),
        lambda: jnp.einsum("ii,jk->jk", sq, a, reduction_order="linear"),
    ):
      with self.assertRaisesRegex(ValueError, "reduction_order="):
        fn()

  def test_ad_vmap_custom_vjp_and_jax_in_one_slide(self):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 4), dtype=jnp.float32)

    # JVP preserves forward linear order on tangents.
    jvp_jaxpr = jax.make_jaxpr(
        lambda a, da: jax.jvp(lambda z: jnp.dot(z, w, reduction_order="linear"), (a,), (da,))
    )(x, x)
    dot_eqns = _eqns_by_name(jvp_jaxpr, "dot_general")
    self.assertLen(dot_eqns, 2)
    self.assertTrue(all(e.params["reduction_order"] == "linear" for e in dot_eqns))

    # Reverse-mode grad leaves backward transpose reductions as reduction_order=None holes.
    def predict(params, inputs):
      for weight, bias in params:
        outputs = jnp.dot(inputs, weight, reduction_order="linear") + bias
        inputs = jnp.maximum(outputs, 0)
      return outputs

    def loss(params, batch):
      inputs, targets = batch
      preds = predict(params, inputs)
      return jnp.sum((preds - targets) ** 2, reduction_order="linear")

    params = [(w, jnp.zeros(4, dtype=jnp.float32)),
              (jnp.ones((4, 4), dtype=jnp.float32), jnp.zeros(4, dtype=jnp.float32))]
    batch = (x, jnp.zeros((4, 4), dtype=jnp.float32))

    with jax.numpy_rank_promotion("allow"):
      grad_jaxpr = jax.make_jaxpr(jax.grad(loss))(params, batch)
      grad_dots = _eqns_by_name(grad_jaxpr, "dot_general")
      self.assertEqual([e.params["reduction_order"] for e in grad_dots[:2]], ["linear", "linear"])
      self.assertTrue(all(e.params.get("reduction_order") is None for e in grad_dots[2:]))
      grad_sums = _eqns_by_name(grad_jaxpr, "reduce_sum")
      self.assertIn(None, [e.params.get("reduction_order") for e in grad_sums])

      g = jax.jit(jax.grad(loss))(params, batch)
      pe_g = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0)))(params, batch)
      self.assertAllClose(g[0][0], jnp.sum(pe_g[0][0], axis=0), atol=1e-5)

    # Grad through jnp.max(..., reduction_order="linear") and custom_vjp pinning backward.
    gm = jax.grad(lambda a: jnp.sum(jnp.max(a, axis=1, reduction_order="linear")))(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4))
    self.assertEqual(gm.shape, (2, 4))
    with self.assertRaises(NotImplementedError):
      jax.vmap(lambda z: jnp.sum(z, axis="x", reduction_order="linear"), axis_name="x")(jnp.ones(4))
    with self.assertRaises(NotImplementedError):
      jax.vmap(lambda z: jnp.max(z, axis="x", reduction_order="linear"), axis_name="x")(jnp.ones(4))

    @jax.custom_vjp
    def pinned_linear(inp, weight):
      return jnp.dot(inp, weight, reduction_order="linear")
    pinned_linear.defvjp(
        lambda inp, weight: (pinned_linear(inp, weight), (inp, weight)),
        lambda res, ct: (jnp.dot(ct, res[1].T, reduction_order="linear"),
                         jnp.dot(res[0].T, ct, reduction_order="linear")))
    cvjp_jaxpr = jax.make_jaxpr(
        jax.grad(lambda inp, weight: jnp.sum(pinned_linear(inp, weight), reduction_order="linear"),
                 argnums=(0, 1)))(x, w)
    self.assertTrue(
        all(e.params["reduction_order"] == "linear" for e in _eqns_by_name(cvjp_jaxpr, "dot_general")))

  def test_trace_time_helpers(self):
    x = jnp.arange(1, 11, dtype=jnp.float32).reshape(2, 5)
    self.assertAllClose(ro.tiled_sum(x, 4, axis=1), jnp.sum(x, axis=1))
    self.assertAllClose(
        ro.tiled_sum(x, 4, axis=1, split="lanes", keepdims=True),
        jnp.sum(x, axis=1, keepdims=True))
    neg_x = -x
    self.assertAllClose(ro.tiled_max(neg_x, 4, axis=1), jnp.max(neg_x, axis=1))
    with self.assertRaises(ValueError):
      ro.tiled_max(neg_x, 4, axis=1, pad=False)

    # xlane_sum / xlane_max jaxpr structure for K < 128, K == 128, K > 128.
    for n, expected_eqns in ((64, 1), (128, 1), (300, 2)):
      v = jnp.ones((2, n), dtype=jnp.float32)
      j_xs = jax.make_jaxpr(lambda z: ro.xlane_sum(z, axis=1))(v)
      s_eqns = _eqns_by_name(j_xs, "reduce_sum")
      self.assertLen(s_eqns, expected_eqns)
      self.assertEqual(s_eqns[-1].params["reduction_order"], ro.Intrinsic("tpu_xlane_128"))
      j_xm = jax.make_jaxpr(lambda z: ro.xlane_max(z, axis=1))(v)
      m_eqns = _eqns_by_name(j_xm, "reduce_max")
      self.assertLen(m_eqns, expected_eqns)
      self.assertEqual(m_eqns[-1].params["reduction_order"], ro.Intrinsic("tpu_xlane_128"))

    # blocked_dot on unaligned K and batched ND contraction with batch dim > contracting dim.
    lhs = jnp.arange(30, dtype=jnp.float32).reshape(2, 3, 5)
    rhs = jnp.arange(40, dtype=jnp.float32).reshape(5, 2, 4)
    dnums = (((2,), (0,)), ((0,), (1,)))
    self.assertAllClose(
        ro.blocked_dot(lhs, rhs, 4, dimension_numbers=dnums),
        lax.dot_general(lhs, rhs, dnums),
        atol=1e-4)
    bd_jaxpr = jax.make_jaxpr(
        lambda l, r: ro.blocked_dot(l, r, 4, dimension_numbers=dnums))(lhs, rhs)
    self.assertLen(_eqns_by_name(bd_jaxpr, "dot_general"), 1)
    self.assertLen(_eqns_by_name(bd_jaxpr, "reduce_sum"), 1)

    # 1-block blocked_dot (K <= bk) skips redundant 1-block reshape + outer reduce_sum.
    for bk_single in (5, 8):
      bd1_jaxpr = jax.make_jaxpr(
          lambda l, r: ro.blocked_dot(
              l, r, bk_single, dimension_numbers=dnums,
              inner_order=ro.Intrinsic("tpu_matmul_bf16"),
              outer_order=ro.Intrinsic("tpu_xlane_128")))(lhs, rhs)
      self.assertLen(_eqns_by_name(bd1_jaxpr, "dot_general"), 1)
      self.assertEmpty(_eqns_by_name(bd1_jaxpr, "reduce_sum"))
      self.assertAllClose(
          ro.blocked_dot(lhs, rhs, bk_single, dimension_numbers=dnums),
          lax.dot_general(lhs, rhs, dnums),
          atol=1e-4)

  def test_unordered_scopes_and_partial_specs(self):
    a = jnp.arange(1, 33, dtype=jnp.float32).reshape(4, 8)
    b = jnp.arange(1, 25, dtype=jnp.float32).reshape(8, 3)
    for ord_val in (ro.Unordered(), ro.Unordered("xla"), ro.Unordered("mosaic")):
      for fn, ref, prim in [
          (lambda x: jnp.sum(x, axis=1, reduction_order=ord_val), jnp.sum(a, axis=1), "reduce_sum"),
          (lambda x: jnp.max(x, axis=1, reduction_order=ord_val), jnp.max(a, axis=1), "reduce_max"),
          (lambda x: jnp.dot(x, b, reduction_order=ord_val), jnp.dot(a, b), "dot_general"),
          (lambda x: jnp.matmul(x, b, reduction_order=ord_val), jnp.matmul(a, b), "dot_general"),
          (lambda x: jnp.einsum("ik,kj->ij", x, b, reduction_order=ord_val), a @ b, "dot_general"),
          (lambda x: jnp.einsum("ij->i", x, reduction_order=ord_val), jnp.sum(a, axis=1), "reduce_sum"),
      ]:
        eqn, = _eqns_by_name(jax.make_jaxpr(fn)(a), prim)
        self.assertEqual(eqn.params["reduction_order"], ord_val)
        self.assertAllClose(fn(a), ref, atol=1e-4)
        self.assertAllClose(jax.jit(fn)(a), ref, atol=1e-4)

    gm = jax.grad(lambda x: jnp.sum(jnp.max(x, axis=1, reduction_order=ro.Unordered("xla"))))(a)
    self.assertAllClose(gm, jax.grad(lambda x: jnp.sum(jnp.max(x, axis=1)))(a))

    inner_u, outer_u = ro.Unordered("mosaic"), ro.Unordered("xla")
    ts_fn = lambda x: ro.tiled_sum(x, 4, axis=1, inner_order=inner_u, outer_order=outer_u)
    ts_eqns = _eqns_by_name(jax.make_jaxpr(ts_fn)(a), "reduce_sum")
    self.assertEqual([e.params["reduction_order"] for e in ts_eqns], [inner_u, outer_u])
    self.assertAllClose(ts_fn(a), jnp.sum(a, axis=1), atol=1e-4)
    self.assertAllClose(jax.jit(ts_fn)(a), jnp.sum(a, axis=1), atol=1e-4)

    bd_fn = lambda x, y: ro.blocked_dot(x, y, 4, inner_order=inner_u, outer_order=outer_u)
    bd_jaxpr = jax.make_jaxpr(bd_fn)(a, b)
    self.assertEqual(_eqns_by_name(bd_jaxpr, "dot_general")[0].params["reduction_order"], inner_u)
    self.assertEqual(_eqns_by_name(bd_jaxpr, "reduce_sum")[0].params["reduction_order"], outer_u)
    self.assertAllClose(bd_fn(a, b), a @ b, atol=1e-4)
    self.assertAllClose(jax.jit(bd_fn)(a, b), a @ b, atol=1e-4)

  def test_absent_vs_zero_padding_signbit_and_validation(self):
    for dt in (jnp.float32, jnp.bfloat16):
      nz = jnp.full((2, 5), -0.0, dtype=dt)
      pos_ones_rhs = jnp.ones((5, 3), dtype=dt)
      pos_ones_lhs = jnp.ones((2, 5), dtype=dt)
      nz_rhs = jnp.full((5, 3), -0.0, dtype=dt)

      # tiled_sum: pad="absent" preserves -0.0; pad="zero" flips to +0.0 (eager & jit).
      for split in ("block", "lanes"):
        ts_abs = lambda a, s=split: ro.tiled_sum(a, 4, axis=1, split=s, pad="absent")
        ts_zero = lambda a, s=split: ro.tiled_sum(a, 4, axis=1, split=s, pad="zero")
        for fn in (ts_abs, jax.jit(ts_abs)):
          out = fn(nz)
          self.assertEqual(out.dtype, dt)
          self.assertTrue(bool(jnp.all(out == 0)))
          self.assertTrue(bool(jnp.all(jnp.signbit(out))))
        for fn in (ts_zero, jax.jit(ts_zero)):
          out = fn(nz)
          self.assertTrue(bool(jnp.all(out == 0)))
          self.assertFalse(bool(jnp.any(jnp.signbit(out))))

      # blocked_dot: (-0.0) * (+0.0) == -0.0 on padded tail for multi-block & 1-block (eager & jit).
      for bk in (4, 8):
        bd_abs = lambda x, y, b=bk, d=dt: ro.blocked_dot(
            x, y, b, preferred_element_type=d, pad="absent")
        bd_zero = lambda x, y, b=bk, d=dt: ro.blocked_dot(
            x, y, b, preferred_element_type=d, pad="zero")
        for lhs_arg, rhs_arg in ((nz, pos_ones_rhs), (pos_ones_lhs, nz_rhs)):
          for fn in (bd_abs, jax.jit(bd_abs)):
            out = fn(lhs_arg, rhs_arg)
            self.assertEqual(out.dtype, dt)
            self.assertTrue(bool(jnp.all(out == 0)))
            self.assertTrue(bool(jnp.all(jnp.signbit(out))))
          for fn in (bd_zero, jax.jit(bd_zero)):
            out = fn(lhs_arg, rhs_arg)
            self.assertTrue(bool(jnp.all(out == 0)))
            self.assertFalse(bool(jnp.any(jnp.signbit(out))))

    # Also exercise >8KiB _linear_dot loop path, mixed int/float, complex, and int under jit.
    big_nz = jnp.full((64, 5), -0.0, dtype=jnp.float32)
    big_ones = jnp.ones((5, 32), dtype=jnp.float32)
    big_out = jax.jit(lambda x, y: ro.blocked_dot(x, y, 4, pad="absent"))(big_nz, big_ones)
    self.assertTrue(bool(jnp.all(jnp.signbit(big_out))))
    long_out = jax.jit(lambda x, y: jnp.dot(x, y, reduction_order="linear"))(
        jnp.full((129, 17), -0.0, dtype=jnp.float32),
        jnp.ones((17, 1), dtype=jnp.float32))
    self.assertTrue(bool(jnp.all(jnp.signbit(long_out))))

    int_lhs = jnp.ones((2, 5), dtype=jnp.int32)
    f32_nz_rhs = jnp.full((5, 3), -0.0, dtype=jnp.float32)
    mixed_out = jax.jit(lambda x, y: ro.blocked_dot(x, y, 4, pad="absent"))(int_lhs, f32_nz_rhs)
    self.assertTrue(bool(jnp.all(jnp.signbit(mixed_out))))
    self.assertArraysEqual(
        ro.tiled_sum(int_lhs, 4, axis=1, pad="absent"),
        jnp.sum(int_lhs, axis=1, dtype=jnp.int32))

    c_nz = jnp.full((2, 5), complex(-0.0, -0.0), dtype=jnp.complex64)
    c_out = jax.jit(lambda a: ro.tiled_sum(a, 4, axis=1, pad="absent"))(c_nz)
    self.assertTrue(bool(jnp.all(jnp.signbit(jnp.real(c_out)) & jnp.signbit(jnp.imag(c_out)))))

    # xlane_sum: K < 128 and K > 128 with f32 and bf16 inputs (eager & jit).
    ro.register_intrinsic(
        "tpu_xlane_128",
        lambda a, *, axes, op: lax.reduce_sum(a, axes, reduction_order="linear"))
    try:
      for dt in (jnp.float32, jnp.bfloat16):
        for k in (5, 130):
          x_nz = jnp.full((2, k), -0.0, dtype=dt)
          xs_abs = lambda a: ro.xlane_sum(a, axis=1, pad="absent")
          xs_zero = lambda a: ro.xlane_sum(a, axis=1, pad="zero")
          for fn in (xs_abs, jax.jit(xs_abs)):
            out = fn(x_nz)
            self.assertTrue(bool(jnp.all(out == 0)))
            self.assertTrue(bool(jnp.all(jnp.signbit(out))))
          for fn in (xs_zero, jax.jit(xs_zero)):
            out = fn(x_nz)
            self.assertTrue(bool(jnp.all(out == 0)))
            self.assertFalse(bool(jnp.any(jnp.signbit(out))))
    finally:
      ro._INTRINSIC_IMPLS.pop("tpu_xlane_128", None)

    # Invalid pad raises ValueError even when K is divisible by tile / bk / 128.
    div_x = jnp.ones((2, 8), dtype=jnp.float32)
    div_y = jnp.ones((8, 3), dtype=jnp.float32)
    div_128 = jnp.ones((2, 128), dtype=jnp.float32)
    unaligned_x = jnp.ones((2, 5), dtype=jnp.float32)
    unaligned_y = jnp.ones((5, 3), dtype=jnp.float32)
    for bad_pad in ("bogus", "", False, True, None, 0):
      for call in (
          lambda p=bad_pad: ro.tiled_sum(div_x, 4, axis=1, pad=p),  # type: ignore[arg-type]
          lambda p=bad_pad: ro.tiled_sum(unaligned_x, 4, axis=1, pad=p),  # type: ignore[arg-type]
          lambda p=bad_pad: ro.xlane_sum(div_128, axis=1, pad=p),  # type: ignore[arg-type]
          lambda p=bad_pad: ro.xlane_sum(unaligned_x, axis=1, pad=p),  # type: ignore[arg-type]
          lambda p=bad_pad: ro.blocked_dot(div_x, div_y, 4, pad=p),  # type: ignore[arg-type]
          lambda p=bad_pad: ro.blocked_dot(unaligned_x, unaligned_y, 4, pad=p),  # type: ignore[arg-type]
      ):
        with self.assertRaises(ValueError):
          call()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
