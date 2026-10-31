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

"""Precision tests for `jax.scipy.stats.truncnorm`."""

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special
import scipy.stats

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]

_PREC = 250


# ---------------------------------------------------------------------------
# High-precision `mpmath` reference implementations (250-bit arithmetic)
# ---------------------------------------------------------------------------


def _erfcx_nonneg(x):
  """Extended-precision `erfcx(x) = e^{x^2} erfc(x)` for `x >= 0`."""
  if x == 0:
    return mpmath.mpf(1)
  if x == mpmath.inf:
    return mpmath.mpf(0)
  if x < 50:
    with mpmath.extraprec(int(x * x * 2) + 32):
      return mpmath.exp(x * x) * mpmath.erfc(x)
  # For `x >= 50`, `2 x^2 >= 5000`, so 40 asymptotic terms give > 300 bits.
  inv_2x2 = 1 / (2 * x * x)
  term = mpmath.mpf(1)
  total = mpmath.mpf(1)
  for k in range(1, 45):
    term *= -(2 * k - 1) * inv_2x2
    total += term
    if abs(term) < mpmath.eps / 4:
      break
  return total / (x * mpmath.sqrt(mpmath.pi))


def _exact_sub(x, y):
  if not (mpmath.isfinite(x) and mpmath.isfinite(y)):
    return x - y
  return mpmath.mpf(0) if x == y else mpmath.fsub(x, y, exact=True)


def _exact_add(x, y):
  if not (mpmath.isfinite(x) and mpmath.isfinite(y)):
    return x + y
  return mpmath.mpf(0) if x == -y else mpmath.fadd(x, y, exact=True)


def _half_sq_diff(x, y):
  return _exact_sub(x, y) * _exact_add(x, y) / 2


def _mpmath_scaled_log_mass(a, b, prec=_PREC):
  """Returns `(x0, ls)` where `log(Phi(b) - Phi(a)) = -x0^2/2 + ls`.

  Handles all interval types: central (`a < 0 < b`), same-side tail
  (`0 <= a < b`), and reflections. Adaptively increases working precision
  to absorb cancellation on narrow intervals.
  """
  # Reflect to `0 <= a' < b'`; `fneg(exact=True)` avoids rounding an unrounded
  # `c + d` passed in at lower `workprec` from `_mpmath_ppf`.
  if b <= 0:
    a, b = mpmath.fneg(b, exact=True), mpmath.fneg(a, exact=True)
  s = mpmath.sqrt(mpmath.mpf(0.5))
  if a < 0:  # Central interval: straddles zero, so `x0 = 0`.
    with mpmath.workprec(prec + 32):
      tail_a = mpmath.exp(-a * a / 2) * _erfcx_nonneg(-a * s)
      tail_b = (
          0
          if b == mpmath.inf
          else mpmath.exp(-b * b / 2) * _erfcx_nonneg(b * s)
      )
      tail = (tail_a + tail_b) / 2
      if tail <= 0.5:
        r = mpmath.log1p(-tail)
      else:
        r = mpmath.log((mpmath.erf(-a * s) + mpmath.erf(b * s)) / 2)
    return mpmath.mpf(0), +r

  # Same-side tail `0 <= a < b`: adaptively increase precision to absorb
  # cancellation on narrow intervals where `b - a << a`.
  width = _exact_sub(b, a)
  extra = 32
  if width != mpmath.inf and a != 0:
    extra += max(0, int(mpmath.mag(a) - mpmath.mag(width)))
  for _ in range(20):
    with mpmath.workprec(prec + extra):
      t1 = _erfcx_nonneg(a * s)
      if b == mpmath.inf:
        val = t1
        break
      t2 = mpmath.exp(-width * (b + a) / 2) * _erfcx_nonneg(b * s)
      val = t1 - t2
      lost = mpmath.mag(t1) - mpmath.mag(val) if val else extra + 64
    if lost + 32 <= extra:
      break
    extra = int(lost) + 64
  else:
    raise RuntimeError(f"_mpmath_scaled_log_mass did not converge: {a}, {b}")
  with mpmath.workprec(prec + 32):
    return a, +mpmath.log(val / 2)


def _mpmath_logpdf(x, a, b, scaled_ab=None):
  if mpmath.isnan(x) or mpmath.isnan(a) or mpmath.isnan(b) or a >= b:
    return mpmath.nan
  if x < a or x > b or mpmath.isinf(x):
    return -mpmath.inf
  with mpmath.workprec(_PREC + 32):
    x0, ls = (
        scaled_ab if scaled_ab is not None else _mpmath_scaled_log_mass(a, b)
    )
    return +(_half_sq_diff(x0, x) - mpmath.log(2 * mpmath.pi) / 2 - ls)


def _mpmath_logcdf(x, a, b, scaled_ab=None):
  if mpmath.isnan(x) or mpmath.isnan(a) or mpmath.isnan(b) or a >= b:
    return mpmath.nan
  if x <= a:
    return -mpmath.inf
  if x >= b:
    return mpmath.mpf(0)
  with mpmath.workprec(_PREC + 32):
    x0_ab, ls_ab = (
        scaled_ab if scaled_ab is not None else _mpmath_scaled_log_mass(a, b)
    )
    x0_ax, ls_ax = _mpmath_scaled_log_mass(a, x)
    logcdf = _half_sq_diff(x0_ab, x0_ax) + (ls_ax - ls_ab)
    if logcdf > -mpmath.log(2):
      x0_xb, ls_xb = _mpmath_scaled_log_mass(x, b)
      logsf = _half_sq_diff(x0_ab, x0_xb) + (ls_xb - ls_ab)
      return +mpmath.log1p(-mpmath.exp(logsf))
    return +logcdf


def _mpmath_ppf(q, a, b, scaled_ab=None):
  """Inverts `truncnorm.cdf` via bracketed Newton iteration in log space.

  Solves `log(Phi(z) - Phi(a)) - log(Phi(b) - Phi(a)) - log(q) = 0`.
  Iterates on `d = z - c` relative to a nearby base point `c` so sub-ULP
  steps from `a` or `b` are not lost to rounding. Two passes: first at
  80-bit precision for a warm start, then at full `_PREC` precision.
  """
  if mpmath.isnan(q) or mpmath.isnan(a) or mpmath.isnan(b):
    return mpmath.nan
  if q < 0 or q > 1 or a >= b:
    return mpmath.nan
  if q == 0:
    return a
  if q == 1:
    return b
  with mpmath.workprec(_PREC + 16):
    if q > mpmath.mpf("0.5"):
      neg_ab = None if scaled_ab is None else (-scaled_ab[0], scaled_ab[1])
      return -_mpmath_ppf(1 - q, -b, -a, scaled_ab=neg_ab)

    x0_ab_full, ls_ab_full = (
        scaled_ab if scaled_ab is not None else _mpmath_scaled_log_mass(a, b)
    )
    log_q_full = mpmath.log(q)

    # Choose initial `(c, d)` with `a < c + d < b`.
    if mpmath.isfinite(a):
      step = mpmath.exp(
          log_q_full + ls_ab_full + _half_sq_diff(a, x0_ab_full)
      ) * mpmath.sqrt(2 * mpmath.pi)
      if step <= (b - a) / 2 and not (a < 0 < b and a + step > 0):
        c, d = a, step
      elif mpmath.isfinite(b) and b - a <= 1:
        c, d = a, (b - a) / 2
      else:
        c = mpmath.mpf(0)
        d = (a + min(b, mpmath.mpf(0))) / 2
    else:  # `a == -inf`: `log Phi(z) = -x0_ab^2 / 2 + ls_ab + log_q`.
      log_phi_z = -x0_ab_full * x0_ab_full / 2 + ls_ab_full + log_q_full
      z0 = (
          -mpmath.sqrt(-2 * log_phi_z)
          if log_phi_z < -2
          else min(mpmath.mpf(0), b - 1)
      )
      if z0 >= b:
        z0 = b - 1
      c = b if (mpmath.isfinite(b) and b - z0 < 1) else mpmath.mpf(0)
      d = z0 - c

    for prec, tol in (
        (80, mpmath.mpf(2) ** -56),
        (_PREC, mpmath.mpf(2) ** -(_PREC // 2)),
    ):
      with mpmath.workprec(prec + 16):
        lo, hi = a - c, b - c
        x0_ab = +x0_ab_full
        ls_ab = +ls_ab_full
        log_q = +log_q_full
        inv_sqrt_2pi = 1 / mpmath.sqrt(2 * mpmath.pi)
        for _ in range(200):
          z = _exact_add(c, d)
          x0_z, ls_z = _mpmath_scaled_log_mass(a, z, prec=prec)
          g = _half_sq_diff(x0_ab, x0_z) + (ls_z - ls_ab) - log_q
          if g == 0:
            break
          if g > 0:
            hi = d
          else:
            lo = d
          slope = mpmath.exp(_half_sq_diff(x0_z, z) - ls_z) * inv_sqrt_2pi
          d_new = d - g / slope if mpmath.isfinite(g) else d
          if (
              mpmath.isfinite(g)
              and abs(d_new - d) <= max(abs(d_new), abs(z)) * tol
          ):
            d = d_new
            break
          if not lo < d_new < hi:
            if mpmath.isfinite(lo) and mpmath.isfinite(hi):
              d_new = (lo + hi) / 2
            elif mpmath.isfinite(hi):
              d_new = hi - 2 * (1 + abs(hi))
            else:
              d_new = lo + 2 * (1 + abs(lo))
          d = d_new
        else:
          raise RuntimeError(f"_mpmath_ppf did not converge: {q}, {a}, {b}")
    return +_exact_add(c, d)


# ---------------------------------------------------------------------------
# Float64 reference wrappers and domain-masking helpers
# ---------------------------------------------------------------------------


def _ref_f64(fn):
  """Wraps a JAX `truncnorm` function as a vectorized `float64` reference.

  `check_unary_precision` uses `mpmath_fn` for `float64` checks but needs a
  fast vectorized reference for lower-precision sweeps (up to 2^20 points).
  SciPy's `truncnorm` loses accuracy on narrow intervals and far tails even in
  `float64`, so we use JAX's own `float64` (verified against `mpmath` in the
  `_float64` tests) as the reference for `bfloat16`/`float16`/`float32`.
  """
  jitted = jax.jit(fn)

  def wrapped(*args: np.ndarray) -> np.ndarray:
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(True), jax.default_device(cpu):
      jax_args = [jnp.asarray(a, dtype=jnp.float64) for a in args]
      return np.array(jitted(*jax_args), dtype=np.float64, copy=True)

  return wrapped


_ref_ppf = _ref_f64(jsp.stats.truncnorm.ppf)
_ref_logpdf = _ref_f64(jsp.stats.truncnorm.logpdf)
_ref_logcdf = _ref_f64(jsp.stats.truncnorm.logcdf)


def _ppf_window_ignored(
    x: np.ndarray, w: float, q0: float = 0.25
) -> np.ndarray:
  """Masks ill-conditioned inputs for `ppf(q0, x, x+w)`.

  Two cases are masked out:
  1. Zero-crossing: when `x < 0 < x + w` and the output `z`
     is near zero, the condition number blows up because a
     tiny rounding in `Phi(z) ~ 0.5` maps to a large
     relative error in `z ~ 0`.
  2. Overflow: `|x| > 0.25 * sqrt(fmax)` overflows `x^2`.
  """
  dt = x.dtype.type
  x64 = x.astype(np.float64)
  b64 = (x + dt(w)).astype(np.float64)
  with np.errstate(all="ignore"):
    z = scipy.stats.truncnorm.ppf(q0, x64, b64)
    mass = scipy.special.ndtr(b64) - scipy.special.ndtr(x64)
    kappa = q0 * mass / (scipy.stats.norm.pdf(z) * np.abs(z))
  near_zero_crossing = (x64 < 0.0) & (b64 > 0.0) & (kappa > 4.0)
  lim = np.sqrt(float(jnp.finfo(x.dtype).max)) * 0.25
  return near_zero_crossing | (np.isfinite(x) & (np.abs(x) > lim))


_WINDOW_POINTS = [
    -37.0, -25.0, -10.0, -8.0, -2.0, -1.5, -1.0, -0.125, -1e-20, 0.0,
    1e-20, 0.5, 1.0, 1.5, 2.0, 8.0, 10.0, 20.0, 25.0, 37.0,
]

_QUANTILE_POINTS = [
    0.0, 1e-30, 1e-20, 1e-12, 1e-6, 1e-4, 0.01, 0.1, 0.25, 0.5,
    0.75, 0.9, 0.99, 1.0 - 1e-4, 1.0 - 1e-6, 1.0 - 1e-12, 1.0,
]


def _check_interval_sweep(
    tc,
    dtype,
    kind: str,
    a0: float,
    b0: float,
    bounds,
):
  """Sweeps the evaluation point `x` across a fixed `[a0, b0]` interval."""
  dt = np.dtype(dtype).type
  a_dt, b_dt = dt(a0), dt(b0)
  a64, b64 = float(a_dt), float(b_dt)
  with mpmath.workprec(_PREC + 32):
    scaled_ab = _mpmath_scaled_log_mass(mpmath.mpf(a64), mpmath.mpf(b64))

  fn = (
      jsp.stats.truncnorm.logpdf
      if kind == "logpdf"
      else jsp.stats.truncnorm.logcdf
  )
  ref_impl = _ref_logpdf if kind == "logpdf" else _ref_logcdf
  mp_impl = _mpmath_logpdf if kind == "logpdf" else _mpmath_logcdf

  @jax.jit
  def jax_fn(x):
    return fn(
        x, jnp.asarray(a_dt, dtype=x.dtype), jnp.asarray(b_dt, dtype=x.dtype)
    )

  pts = [
      a64 + t * (b64 - a64)
      for t in (0.0, 1e-12, 1e-6, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9,
                0.99, 1.0 - 1e-4, 1.0 - 1e-6, 1.0 - 1e-12, 1.0)
  ]
  util.check_unary_precision(
      tc,
      jax_fn,
      lambda x64: ref_impl(
          x64, np.full_like(x64, a64), np.full_like(x64, b64)
      ),
      lambda x_mp: mp_impl(
          x_mp, mpmath.mpf(a64), mpmath.mpf(b64), scaled_ab=scaled_ab
      ),
      dtype,
      bounds=bounds,
      input_ftz=[("gpu", False)],
      check_signed_zeros=False,
      max_samples=4000 if dtype == f64 else None,
      interesting_points=pts,
  )


# ---------------------------------------------------------------------------
# Test classes (one class per shard under `util.ClassShardedTestLoader`)
# ---------------------------------------------------------------------------


@jtu.thread_unsafe_test_class()
class TruncnormLogPdfTest(jtu.JaxTestCase):
  """Precision and dtype tests for `truncnorm.logpdf`."""

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logpdf_narrow(self, dtype):
    bounds = [
        ("cpu", {bf16: 0.5, f16: 0.5, f32: (1.0, 2.5), f64: 2.5}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5, f64: 3.5}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5}),
    ]
    _check_interval_sweep(self, dtype, "logpdf", 1.25, 1.375, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logpdf_narrow_tail(self, dtype):
    # Narrow interval `[8.0, 8.125]`: quadrature + `exp(-32)` cancellation.
    bounds = [
        ("cpu", {bf16: 0.5, f16: 0.5, f32: (1.0, 2.5), f64: 3.0}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5, f64: 4.0}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5}),
    ]
    _check_interval_sweep(self, dtype, "logpdf", 8.0, 8.125, bounds)

  @config.explicit_x64_dtypes("allow")
  def test_explicit_x64_dtypes(self):
    # Verifies explicit float64 inputs stay float64 with `enable_x64=False`.
    if jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    a = jnp.array([-1.0, 1.25, 0.5, 8.0], dtype=jnp.float64)
    b = jnp.array([1.5, 1.375, 0.625, 10.0], dtype=jnp.float64)
    x = jnp.array([0.2, 1.3, 0.55, 8.5], dtype=jnp.float64)
    q = jnp.array([0.3, 0.25, 0.02, 0.7], dtype=jnp.float64)

    @jax.jit
    def eval_all(x, q, a, b):
      grad_loss = lambda x_, q_, a_, b_: (
          jnp.sum(jsp.stats.truncnorm.logpdf(x_, a_, b_))
          + jnp.sum(jsp.stats.truncnorm.logcdf(x_, a_, b_))
          + jnp.sum(jsp.stats.truncnorm.ppf(q_, a_, b_))
      )
      return (
          jsp.stats.truncnorm.logpdf(x, a, b),
          jsp.stats.truncnorm.logcdf(x, a, b),
          jsp.stats.truncnorm.logsf(x, a, b),
          jsp.stats.truncnorm.ppf(q, a, b),
          jsp.stats.truncnorm.isf(q, a, b),
          *jax.grad(grad_loss, argnums=(0, 1, 2, 3))(x, q, a, b),
      )

    with jax.enable_x64(True):
      expected = eval_all(x, q, a, b)
    with jax.enable_x64(False):
      actual = eval_all(x, q, a, b)
    for act, exp in zip(actual, expected):
      self.assertEqual(act.dtype, jnp.float64)
      self.assertArraysEqual(act, exp)


@jtu.thread_unsafe_test_class()
class TruncnormLogCdfTest(jtu.JaxTestCase):
  """Precision tests for `truncnorm.logcdf`."""

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logcdf_narrow(self, dtype):
    bounds = [
        ("cpu", {bf16: 0.5, f16: 0.5, f32: (1.5, 3.0), f64: 3.0}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 4.0, f64: 4.0}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 4.0}),
    ]
    _check_interval_sweep(self, dtype, "logcdf", 1.25, 1.375, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logcdf_wide(self, dtype):
    # Exercises narrow `[a, x]` near `a`, wide central, and the
    # `log1p(-sf)` branch for `logcdf`.
    bounds = [
        ("cpu", {bf16: (0.5, 1.0), f16: 0.5, f32: (3.5, 5.5), f64: 4.5}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 6.0, f64: 5.5}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 6.0}),
    ]
    _check_interval_sweep(self, dtype, "logcdf", -1.0, 2.0, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logcdf_tail(self, dtype):
    # Wide tail `[8.0, 10.0]`: exercises quadrature `[a, x]` near `a`, `erfcx`
    # tail mass, and `log1p(-sf)` near `b`. Near `x = 10`, rounding
    # `-(x^2 - 8^2)/2 ~= -18` in working precision introduces ~1 ULP error in
    # the exponent, which `exp` scales by 18 into ~19-31 ULPs in `log1p(-sf)`.
    bounds = [
        (
            "cpu",
            {bf16: (0.5, 1.0), f16: (0.5, 1.0), f32: (23.5, 26.5), f64: 20.5},
        ),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 32.5, f64: 24.5}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 31.5}),
    ]
    _check_interval_sweep(self, dtype, "logcdf", 8.0, 10.0, bounds)

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_logpdf_and_logcdf_special_points(self, dtype):
    # Tests logpdf and logcdf on challenging micro-intervals: odd-ULP widths
    # (21 * ulp, testing quadrature without midpoint rounding error),
    # subnormal-width intervals, and very narrow intervals near large `a`.
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    dt = np.dtype(dtype).type
    ulp_100 = float(np.spacing(dt(100.0)))
    tiny = float(jnp.finfo(dtype).tiny)
    points = [
        (1.2 + 2e-6, 1.2, 1.2 + 1e-5),
        (1.2 + 8e-6, 1.2, 1.2 + 1e-5),
        (19.1, 19.0, 19.35),
        (100.0 + 5.0 * ulp_100, 100.0, 100.0 + 21.0 * ulp_100),
        (100.0 + 16.0 * ulp_100, 100.0, 100.0 + 21.0 * ulp_100),
        (-100.0 - 16.0 * ulp_100, -100.0 - 21.0 * ulp_100, -100.0),
        (2.0 * tiny, 0.0, 8.0 * tiny),
    ]
    if dtype == f64:
      ulp_1e4 = float(np.spacing(np.float64(1e4)))
      points.extend([
          (1e4 + 5.0 * ulp_1e4, 1e4, 1e4 + 21.0 * ulp_1e4),
          (1e4 + 16.0 * ulp_1e4, 1e4, 1e4 + 21.0 * ulp_1e4),
      ])
    x_arr, a_arr, b_arr = (
        jnp.array([p[i] for p in points], dtype=dtype) for i in range(3)
    )
    for jax_fn, mp_fn in (
        (jsp.stats.truncnorm.logpdf, _mpmath_logpdf),
        (jsp.stats.truncnorm.logcdf, _mpmath_logcdf),
    ):
      util.check_precision_at_points(
          self, jax.jit(jax_fn), mp_fn, x_arr, a_arr, b_arr, max_ulp=5.0
      )


@jtu.thread_unsafe_test_class()
class TruncnormPpfQuantileTest(jtu.JaxTestCase):
  """Precision tests sweeping quantile `q` for `truncnorm.ppf`."""

  def _check_quantile_sweep(self, dtype, a0: float, b0: float, bounds):
    dt = np.dtype(dtype).type
    a_dt, b_dt = dt(a0), dt(b0)
    a64, b64 = float(a_dt), float(b_dt)
    with mpmath.workprec(_PREC + 16):
      scaled_ab = _mpmath_scaled_log_mass(mpmath.mpf(a64), mpmath.mpf(b64))

    @jax.jit
    def jax_fn(q):
      return jsp.stats.truncnorm.ppf(
          q, jnp.asarray(a_dt, dtype=q.dtype), jnp.asarray(b_dt, dtype=q.dtype)
      )

    util.check_unary_precision(
        self,
        jax_fn,
        lambda q64: _ref_ppf(
            q64, np.full_like(q64, a64), np.full_like(q64, b64)
        ),
        lambda q_mp: _mpmath_ppf(
            q_mp, mpmath.mpf(a64), mpmath.mpf(b64), scaled_ab=scaled_ab
        ),
        dtype,
        bounds=bounds,
        input_ftz=[("gpu", False)],
        check_signed_zeros=False,
        max_samples=2500 if dtype == f64 else None,
        interesting_points=_QUANTILE_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_wide(self, dtype):
    # Wide interval `[0.5, 2.0]`: tests global `ndtri_exp` path and
    # endpoint local step.
    bounds = [
        ("cpu", {bf16: 0.5, f16: 0.5, f32: (3.0, 4.0), f64: 3.5}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 5.0, f64: 4.5}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 5.0}),
    ]
    self._check_quantile_sweep(dtype, 0.5, 2.0, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_narrow(self, dtype):
    # Narrow interval `[1.25, 1.375]`: tests Gauss-Legendre mass +
    # local endpoint step.
    bounds = [
        ("cpu", {bf16: 0.5, f16: 0.5, f32: (1.5, 2.5), f64: 2.0}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5, f64: 3.0}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5}),
    ]
    self._check_quantile_sweep(dtype, 1.25, 1.375, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_tail(self, dtype):
    # Far right tail `[8.0, 10.0]`: tests `erfcx`-based mass + `ndtri_exp`.
    bounds = [
        ("cpu", {bf16: 0.5, f16: 1.0, f32: (1.5, 2.5), f64: 2.0}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5, f64: 3.0}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5}),
    ]
    self._check_quantile_sweep(dtype, 8.0, 10.0, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_quantile_semi_infinite(self, dtype):
    # Semi-infinite interval `[3.0, +inf)`: tests unbounded upper limit.
    bounds = [
        ("cpu", {bf16: 0.5, f16: 0.5, f32: (1.5, 2.5), f64: 2.5}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5, f64: 3.5}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 3.5}),
    ]
    self._check_quantile_sweep(dtype, 3.0, np.inf, bounds)


@jtu.thread_unsafe_test_class()
class TruncnormPpfLocationTest(jtu.JaxTestCase):
  """Precision tests sweeping location `x` for `ppf(0.25, x, x + w)`."""

  def _check_window_sweep(self, dtype, w: float, bounds, q0: float = 0.25):
    dt = np.dtype(dtype).type
    q_dt, w_dt = dt(q0), dt(w)
    q64 = float(q_dt)

    @jax.jit
    def jax_fn(x):
      b = lax.optimization_barrier(x + jnp.asarray(w_dt, dtype=x.dtype))
      return jsp.stats.truncnorm.ppf(jnp.asarray(q_dt, dtype=x.dtype), x, b)

    def ref_fn(x64):
      b64 = (x64.astype(dt) + w_dt).astype(np.float64)
      return _ref_ppf(np.full_like(x64, q64), x64, b64)

    def mpmath_fn(x_mp):
      b_mp = mpmath.mpf(float(dt(float(x_mp)) + w_dt))
      return _mpmath_ppf(mpmath.mpf(q64), x_mp, b_mp)

    util.check_unary_precision(
        self,
        jax_fn,
        ref_fn,
        mpmath_fn,
        dtype,
        bounds=bounds,
        ignore_inputs=lambda x: _ppf_window_ignored(x, w, q0),
        input_ftz=[("gpu", False)],
        check_signed_zeros=False,
        max_samples=4000 if dtype == f64 else None,
        interesting_points=_WINDOW_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_location_narrow_window(self, dtype):
    bounds = [
        ("cpu", {bf16: 0.5, f16: 0.5, f32: (5.5, 7.0), f64: 3.0}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 8.0, f64: 4.5}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 8.0}),
    ]
    self._check_window_sweep(dtype, 0.125, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_ppf_location_wide_window(self, dtype):
    # Near the edge of the unmasked zero-crossing region (condition number
    # `kappa <= 4`, e.g. `x ~= -0.524`, `z ~= -0.106`), each ULP of error
    # in `log Phi(z)` amplifies to ~8-13 ULPs in `z`, so ~2 ULPs in the
    # log-CDF maps to ~17-25 ULPs in the PPF output.
    bounds = [
        ("cpu", {bf16: 0.5, f16: 1.0, f32: (23.5, 25.5), f64: 18.0}),
        ("gpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 28.0, f64: 20.0}),
        ("tpu", {bf16: (0.5, 1.5), f16: (0.5, 1.5), f32: 28.0}),
    ]
    self._check_window_sweep(dtype, 2.0, bounds)

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_ppf_special_points(self, dtype):
    # Specific challenging inputs: odd-ULP narrow spans, subnormal widths,
    # semi-infinite intervals, and tiny quantiles near extreme tails.
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    dt = np.dtype(dtype).type
    ulp_100 = float(np.spacing(dt(100.0)))
    tiny = float(jnp.finfo(dtype).tiny)
    points = [
        (0.5, -2.0, -0.5),
        (0.5, 0.5, 2.0),
        (0.1, 1.2, 1.2 + 1e-5),
        (0.9, 1.2, 1.2 + 1e-5),
        (0.5, 19.0, 19.35),
        (0.5, 20.0, 25.0),
        (0.5, -25.0, -20.0),
        (0.75, 100.0, 100.0 + 21.0 * ulp_100),
        (0.5, 0.0, 5e-21),
        (0.25, -4e-21, 6e-21),
        (0.25, -np.inf, -3.0),
        (0.75, 3.0, np.inf),
        (0.25, -np.inf, np.inf),
        (tiny, -20.0 if dtype == f32 else -40.0, 0.0),
    ]
    if dtype == f64:
      ulp_1e6 = float(np.spacing(np.float64(1e6)))
      points.extend([
          (0.75, -38.308410422147944, -38.201845012464595),
          (0.75, 1e6, 1e6 + 21.0 * ulp_1e6),
          (0.5, 0.0, 1e-300),
      ])
    q_arr, a_arr, b_arr = (
        jnp.array([p[i] for p in points], dtype=dtype) for i in range(3)
    )
    util.check_precision_at_points(
        self,
        jax.jit(jsp.stats.truncnorm.ppf),
        _mpmath_ppf,
        q_arr,
        a_arr,
        b_arr,
        max_ulp=5.0,
    )


util.register_benchmark(
    jsp.stats.truncnorm.ppf, name="truncnorm_ppf", nargs=3
)
util.register_benchmark(
    jsp.stats.truncnorm.logcdf, name="truncnorm_logcdf", nargs=3
)


if __name__ == "__main__":
  util.main()
