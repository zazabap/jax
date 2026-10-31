# Copyright 2022 The JAX Authors.
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

import numpy as np

from jax._src import api
from jax._src import dtypes
from jax._src import lax
from jax._src.lax.lax import AccuracyMode
from jax._src import numpy as jnp
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.special import erfcx, log_ndtr, ndtr, ndtri_exp
from jax._src.typing import Array, ArrayLike


_HALF_LOG_2PI = float(0.5 * np.log(2.0 * np.pi))
_LOG_2 = float(np.log(2.0))
_SQRT_2 = float(np.sqrt(2.0))
_INV_SQRT_2 = float(np.sqrt(0.5))

# When `b - a <= 1.48` and the interval midpoint is moderate, 11-point
# Gauss-Legendre quadrature gives < 0.5 ULP error in float64. Beyond that
# the endpoints' CDFs are well-separated and the wide-interval formulas
# suffice (< 0.25 bits lost).
#
# In `ppf`, when the quantile is near an endpoint (normalized residual
# `r_0 <= 0.08` and `|a_0 * r_0| <= 0.08`), a 6th-order series + one
# Newton step resolves the step to < 2 ULPs in float64.
_QUAD_MAX_WIDTH = 1.48
_QUAD_MAX_ABS_W = 2.0
_LOCAL_STEP_MAX = 0.08
_LOG_LOCAL_STEP_MAX = float(np.log(_LOCAL_STEP_MAX))

# Positive-half nodes and weights for the 11-point Gauss-Legendre rule on
# `[-1, 1]`, folded onto `[0, 1]` using the even symmetry of the integrand.
# The center node (`u = 0`) drops out because the integrand is zero there.
_GL11_POS_NODES = (
  0.26954315595234496,
  0.5190961292068118,
  0.7301520055740494,
  0.8870625997680953,
  0.978228658146057,
)
_GL11_POS_WEIGHTS = (
  0.26280454451024665,
  0.23319376459199048,
  0.18629021092773426,
  0.1255803694649046,
  0.05566856711617366,
)

# Nodes and weights of the 4-point Gauss-Legendre rule on `[0, 1]`.
_GL4_S1 = float(np.sqrt(3.0 / 7.0 - (2.0 / 7.0) * np.sqrt(6.0 / 5.0)))
_GL4_S2 = float(np.sqrt(3.0 / 7.0 + (2.0 / 7.0) * np.sqrt(6.0 / 5.0)))
_GL4_NODES = (
  0.5 * (1.0 - _GL4_S2),
  0.5 * (1.0 - _GL4_S1),
  0.5 * (1.0 + _GL4_S1),
  0.5 * (1.0 + _GL4_S2),
)
_GL4_W1 = float((18.0 + np.sqrt(30.0)) / 72.0)
_GL4_W2 = float((18.0 - np.sqrt(30.0)) / 72.0)
_GL4_WEIGHTS = (_GL4_W2, _GL4_W1, _GL4_W1, _GL4_W2)

# Coefficients for a 6th-order bivariate series that inverts
# `integral_0^delta e^{-a_0 t - t^2/2} dt = r_0` for `delta`, used when
# `ppf` is near an endpoint. Grouped by powers of `r_0^2`; each row is
# an ascending-degree polynomial in `z_0 = a_0 * r_0`.
_LOCAL_STEP_COEFFS = (
  (1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0, 1.0 / 6.0),
  (1.0 / 6.0, 7.0 / 24.0, 23.0 / 60.0, 163.0 / 360.0),
  (7.0 / 120.0, 127.0 / 720.0),
)


def _half_sq_diff(x: Array | float, y: Array) -> Array:
  """Evaluates `(x^2 - y^2) / 2` without cancellation."""
  # Halve before adding/subtracting so `x + y` doesn't overflow to `inf`
  # when `|x|, |y| > fmax / 2` (which would produce `0 * inf = NaN`).
  return 2.0 * (0.5 * x - 0.5 * y) * (0.5 * x + 0.5 * y)


def _mask_pair(
  cond: Array, a: Array, b: Array, fill_a: float = -1.0, fill_b: float = 1.0
) -> tuple[Array, Array]:
  return jnp.where(cond, a, fill_a), jnp.where(cond, b, fill_b)


def _clamp_preserve_grad(x: Array, lo: Array, hi: Array) -> Array:
  """Clamps `x` to `[lo, hi]` without `jnp.clip`'s halved gradients at ties."""
  return jnp.where(x < lo, lo, jnp.where(x > hi, hi, x))


def _promote_and_broadcast(
  name: str, *args: ArrayLike
) -> tuple[tuple[Array, ...], np.dtype]:
  promoted = promote_args_inexact(name, *args)
  orig_dtype = promoted[0].dtype
  if dtypes.finfo(orig_dtype).bits < 32:
    promoted = tuple(v.astype(np.float32) for v in promoted)
  return tuple(jnp.broadcast_arrays(*promoted)), orig_dtype


def _sanitize_support(
  a: Array, b: Array, loc: Array, scale: Array
) -> tuple[Array, Array, Array, Array]:
  """Masks invalid `(a, b, scale)` parameters to safe finite values."""
  valid_params = (a < b) & (scale > 0) & ~jnp.isnan(loc)
  a_safe, b_safe = _mask_pair(valid_params, a, b)
  scale_safe = jnp.where(scale > 0, scale, 1.0)
  return valid_params, a_safe, b_safe, scale_safe


def _logaddexp(x1: Array, x2: Array) -> Array:
  """Evaluates `log(exp(x1) + exp(x2))` with high accuracy.

  Guards the case where both inputs are `-inf` (e.g. when both
  `log_ndtr` and `log_mass` underflow) so that reverse-mode AD sees `0`
  instead of `NaN`.
  """
  same_inf = jnp.isinf(x1) & (x1 == x2)
  ge = x1 >= x2
  amax = jnp.where(ge, x1, x2)
  d_lo, d_hi = _mask_pair(~same_inf, jnp.where(ge, x2, x1), amax, 0.0, 0.0)
  l1p = lax.log1p(
    lax.exp(d_lo - d_hi, accuracy=AccuracyMode.HIGHEST),
    accuracy=AccuracyMode.HIGHEST,
  )
  return jnp.where(same_inf, x1, amax + l1p)


def _gauss_mass_quad_corr(h: Array, w: Array) -> Array:
  r"""Quadrature correction for `integral_a^b e^{-x^2/2} dx`.

  For narrow intervals, writes the integral as
  `(b - a) * e^{-m^2/2} * (1 + correction)` where `m = (a + b)/2`
  and `h = (b - a)/2`. Folding `[-1, 1]` onto `[0, 1]` averages
  the `+u` and `-u` contributions via `cosh`. Returns just the
  correction term so callers can pass it to `log1p`.
  """
  acc = jnp.zeros_like(h)
  neg_half_h2 = -0.5 * jnp.square(h)
  for u, wt in zip(_GL11_POS_NODES, _GL11_POS_WEIGHTS):
    exp_a_m1 = lax.expm1(
      (u * u) * neg_half_h2, accuracy=AccuracyMode.HIGHEST
    )
    exp_b_m1 = lax.expm1(u * w, accuracy=AccuracyMode.HIGHEST)
    # Identity: `e^A cosh(B) - 1 =
    #   expm1(A) + e^A * expm1(B)^2 / (2 * (1 + expm1(B)))`.
    term = (
      exp_a_m1
      + 0.5 * (1.0 + exp_a_m1) * jnp.square(exp_b_m1) / (1.0 + exp_b_m1)
    )
    acc = acc + wt * term
  return acc


def _scaled_gauss_mass_wide(a: Array, b: Array) -> tuple[Array, Array]:
  """Log Gaussian mass for intervals too wide for quadrature.

  Returns `(x0, log_scaled)` where
  `log(Phi(b) - Phi(a)) = -0.5 * x0^2 + log_scaled`.

  Three branches:
  - Central (`a < 0 < b`): uses `log1p(-(ndtr(a) + ndtr(-b)))` (no
    cancellation because `ndtr(a) + ndtr(-b) < 0.57` when `b - a > 1.48`).
  - Near same-side tail (e.g. `0 <= a < b`): uses `ndtr(-a) - ndtr(-b)`.
  - Far same-side tail (`a > sqrt(2)`): uses `erfcx` and factors out
    the leading `exp(-0.5 * a^2)`.
  """
  is_central = (a < 0) & (b > 0)
  is_left = b <= 0
  alpha, beta = _mask_pair(
    ~is_central,
    jnp.where(is_left, -b, a),
    jnp.where(is_left, -a, b),
    0.0,
    2.0,
  )
  is_far = ~is_central & (alpha > _SQRT_2)

  # Share one pair of `ndtr` calls across central and near-tail,
  # masking the unused branch so `a == -b` doesn't produce `log(0)`.
  u1 = jnp.where(is_central, a, jnp.where(is_far, 0.0, -alpha))
  u2 = jnp.where(is_central, -b, jnp.where(is_far, -2.0, -beta))
  p1, p2 = ndtr(u1), ndtr(u2)
  log_central = lax.log1p(
    -jnp.where(is_central, p1 + p2, 0.5), accuracy=AccuracyMode.HIGHEST
  )
  log_near = lax.log(
    jnp.where(is_central | is_far, 0.5, p1 - p2),
    accuracy=AccuracyMode.HIGHEST,
  )

  has_upper = is_far & jnp.isfinite(beta)
  alpha_far = jnp.where(is_far, alpha, 2.0)
  beta_far = jnp.where(has_upper, beta, 4.0)
  # Clamp `(beta^2 - alpha^2) / 2` so `exp`'s gradient doesn't overflow.
  max_exp = float(np.log(float(dtypes.finfo(a.dtype).max)))
  delta_sq = jnp.minimum(
    _half_sq_diff(beta_far, jnp.where(has_upper, alpha_far, 2.0)), max_exp
  )
  upper_term = jnp.where(
    has_upper,
    lax.exp(-delta_sq, accuracy=AccuracyMode.HIGHEST)
    * erfcx(beta_far * _INV_SQRT_2),
    0.0,
  )
  log_far = lax.log(
    0.5 * (erfcx(alpha_far * _INV_SQRT_2) - upper_term),
    accuracy=AccuracyMode.HIGHEST,
  )

  x0 = jnp.where(is_far, jnp.where(is_left, -alpha_far, alpha_far), 0.0)
  log_scaled = jnp.where(
    is_central, log_central, jnp.where(is_far, log_far, log_near)
  )
  return x0, log_scaled


def _scaled_gauss_mass_with_quad(
  a: Array, b: Array
) -> tuple[Array, Array, Array, Array, Array]:
  """Gaussian mass decomposition.

  Decomposes `Phi(b) - Phi(a)` as
  `exp(-0.5 * x0^2) * width * exp(log_scaled)`.

  Returns `(x0, width, log_scaled, quad_integral, use_quad)`.

  On narrow intervals (`use_quad`), uses quadrature:
  `width = b - a` and `quad_integral = (b - a) * (1 + correction)`.
  On wide intervals delegates to `_scaled_gauss_mass_wide`
  with `width = 1`.
  """
  is_ordered = a < b
  narrow_width = is_ordered & ((b - a) <= _QUAD_MAX_WIDTH)
  a_narrow, b_narrow = _mask_pair(narrow_width, a, b, 0.0, 1.0)
  # `|m * h| = |(b^2 - a^2) / 4| = 0.5 * |_half_sq_diff(b, a)|`.
  use_quad = narrow_width & (
    0.5 * jnp.abs(_half_sq_diff(b_narrow, a_narrow)) <= _QUAD_MAX_ABS_W
  )

  a_quad, b_quad = _mask_pair(use_quad, a_narrow, b_narrow, 0.0, 1.0)
  dx = b_quad - a_quad
  h = 0.5 * dx
  w = -(a_quad + h) * h
  corr = _gauss_mass_quad_corr(h, w)
  # Factor out `exp(-0.5 * a_quad^2)` instead of the midpoint `m`:
  # when `b - a` is an odd number of ULPs, `m` has a rounding error that
  # gets amplified by `-0.5 * m^2`. Writing `-0.5 * m^2 = -0.5 * a^2 + shift`
  # keeps the correction small.
  shift = -h * (a_quad + 0.5 * h)
  log_scaled_quad = -_HALF_LOG_2PI + (
    shift + lax.log1p(corr, accuracy=AccuracyMode.HIGHEST)
  )
  quad_integral = dx * (1.0 + corr)

  a_wide, b_wide = _mask_pair(is_ordered & ~use_quad, a, b)
  x0_wide, log_scaled_wide = _scaled_gauss_mass_wide(a_wide, b_wide)

  x0 = jnp.where(use_quad, a_quad, x0_wide)
  width = jnp.where(use_quad, dx, 1.0)
  log_scaled = jnp.where(use_quad, log_scaled_quad, log_scaled_wide)
  return x0, width, log_scaled, quad_integral, use_quad


def _scaled_gauss_mass(a: Array, b: Array) -> tuple[Array, Array, Array]:
  """Returns `(x0, width, log_scaled)` for `Phi(b) - Phi(a)`."""
  x0, width, log_scaled, _, _ = _scaled_gauss_mass_with_quad(a, b)
  return x0, width, log_scaled


def _scaled_mass_ratio(
  num: tuple[Array, Array, Array], den: tuple[Array, Array, Array]
) -> tuple[Array, Array]:
  """Log-space ratio of two Gaussian masses."""
  x0_n, w_n, ls_n = num
  x0_d, w_d, ls_d = den
  return w_n / w_d, _half_sq_diff(x0_d, x0_n) + (ls_n - ls_d)


def _ppf_local_step(a0: Array, r0: Array) -> Array:
  r"""Near-endpoint PPF solver.

  Finds `delta` satisfying
  `integral_0^delta e^{-a_0 t - t^2/2} dt = r_0`.

  When the target quantile is close to an endpoint, global CDF inversion
  loses the small step `delta` in the low bits of a large CDF value.
  A 6th-order series gives the initial estimate, and one 4-point
  Gauss-Legendre Newton step refines it to < 2 ULPs in float64.
  """
  # Initial estimate from bivariate series.
  z0 = a0 * r0
  row_coeffs = [
    lax.polynomial(z0, row) for row in _LOCAL_STEP_COEFFS
  ]
  x0 = r0 * lax.polynomial(r0 * r0, row_coeffs)
  # Newton correction: `f(x0) = x0*(1 + quad_avg) - r0`, `f'(x0)`.
  quad_avg = jnp.zeros_like(x0)
  for u, wt in zip(_GL4_NODES, _GL4_WEIGHTS):
    t = u * x0
    quad_avg = quad_avg + wt * lax.expm1(
      -t * (a0 + 0.5 * t), accuracy=AccuracyMode.HIGHEST
    )
  fx = (x0 - r0) + x0 * quad_avg
  dfx = lax.exp(
    -x0 * (a0 + 0.5 * x0), accuracy=AccuracyMode.HIGHEST
  )
  return x0 - fx / dfx


@api.jit
def logpdf(
  x: ArrayLike,
  a: ArrayLike,
  b: ArrayLike,
  loc: ArrayLike = 0,
  scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal log probability distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logpdf``.

  The truncated normal probability distribution is given by

  .. math::

     f(x, a, b) = \begin{cases}
       \frac{1}{\sqrt{2\pi}(\Phi(b) - \Phi(a))}e^{-x^2/2} & a \le x \le b \\
       0 & \mathrm{otherwise}
     \end{cases}

  where :math:`\Phi` is the standard normal CDF and :math:`a` and :math:`b` are
  effectively specified in number of standard deviations from the centroid. JAX
  uses the scipy nomenclature of ``loc`` for the centroid and ``scale`` for the
  standard deviation.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  (x, a, b, loc, scale), orig_dtype = _promote_and_broadcast(
    "truncnorm.logpdf", x, a, b, loc, scale
  )
  valid_params, a_safe, b_safe, scale_safe = _sanitize_support(a, b, loc, scale)
  z = lax.div(lax.sub(x, loc), scale_safe)
  in_support = valid_params & (z >= a) & (z <= b) & jnp.isfinite(z)
  z_safe = jnp.where(in_support, z, 0.0)
  x0, width, log_scaled = _scaled_gauss_mass(a_safe, b_safe)
  # `(x0^2 - z^2) / 2` cancels the leading Gaussian factor when `z`
  # is near `x0`.
  val = (
    _half_sq_diff(x0, z_safe)
    - _HALF_LOG_2PI
    - lax.log(scale_safe, accuracy=AccuracyMode.HIGHEST)
    - (lax.log(width, accuracy=AccuracyMode.HIGHEST) + log_scaled)
  )
  val = jnp.where(in_support, val, -np.inf)
  val = jnp.where(valid_params & ~jnp.isnan(z), val, np.nan)
  return val.astype(orig_dtype)


@api.jit
def pdf(
  x: ArrayLike,
  a: ArrayLike,
  b: ArrayLike,
  loc: ArrayLike = 0,
  scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal probability distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``pdf``.

  The truncated normal probability distribution is given by

  .. math::

     f(x, a, b) = \begin{cases}
       \frac{1}{\sqrt{2\pi}(\Phi(b) - \Phi(a))}e^{-x^2/2} & a \le x \le b \\
       0 & \mathrm{otherwise}
     \end{cases}

  where :math:`\Phi` is the standard normal CDF and :math:`a` and :math:`b` are
  effectively specified in number of standard deviations from the centroid. JAX
  uses the scipy nomenclature of ``loc`` for the centroid and ``scale`` for the
  standard deviation.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return lax.exp(logpdf(x, a, b, loc, scale))


@api.jit
def logsf(
  x: ArrayLike,
  a: ArrayLike,
  b: ArrayLike,
  loc: ArrayLike = 0,
  scale: ArrayLike = 1,
) -> Array:
  """Truncated normal distribution log survival function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logsf``

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the cumulative distribution function,
  :func:`jax.scipy.stats.truncnorm.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  x, a, b, loc, scale = promote_args_inexact(
    "truncnorm.logsf", x, a, b, loc, scale
  )
  return logcdf(-x, -b, -a, -loc, scale)


@api.jit
def sf(
  x: ArrayLike,
  a: ArrayLike,
  b: ArrayLike,
  loc: ArrayLike = 0,
  scale: ArrayLike = 1,
) -> Array:
  """Truncated normal distribution survival function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``sf``

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the cumulative distribution function,
  :func:`jax.scipy.stats.truncnorm.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return lax.exp(logsf(x, a, b, loc, scale))


@api.jit
def logcdf(
  x: ArrayLike,
  a: ArrayLike,
  b: ArrayLike,
  loc: ArrayLike = 0,
  scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf} = \int_{-\infty}^x f_{pdf}(y) \mathrm{d}y

  where here :math:`f_{pdf}` is the probability distribution function,
  :func:`jax.scipy.stats.truncnorm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  (x, a, b, loc, scale), orig_dtype = _promote_and_broadcast(
    "truncnorm.logcdf", x, a, b, loc, scale
  )
  valid_params, a_safe, b_safe, scale_safe = _sanitize_support(a, b, loc, scale)
  z = lax.div(lax.sub(x, loc), scale_safe)
  in_open = valid_params & (z > a_safe) & (z < b_safe)
  z_in = jnp.where(in_open, z, 0.0)

  # Evaluate mass for `[a, b]`, `[a, z]`, and `[z, b]` independently so
  # each sub-interval picks the best regime (quadrature vs. wide).
  mass_ab = _scaled_gauss_mass(a_safe, b_safe)
  w_cdf, log_scaled_cdf = _scaled_mass_ratio(
    _scaled_gauss_mass(a_safe, z_in), mass_ab
  )
  w_sf, log_scaled_sf = _scaled_mass_ratio(
    _scaled_gauss_mass(z_in, b_safe), mass_ab
  )
  logcdf_in = log_scaled_cdf + lax.log(w_cdf, accuracy=AccuracyMode.HIGHEST)
  logsf_in = log_scaled_sf + lax.log(w_sf, accuracy=AccuracyMode.HIGHEST)

  # In the upper half of `[a, b]`, use `log(1 - sf)` instead of `log(cdf)`
  # to avoid subtracting two nearly equal log-masses. Mask the unused
  # branch: in the left tail `sf` rounds to `1.0` and `log1p(-1.0)` has a
  # `NaN` derivative in reverse-mode AD.
  use_sf = in_open & (logcdf_in > -_LOG_2) & (logsf_in < 0.0)
  sf_safe = jnp.where(
    use_sf,
    w_sf
    * lax.exp(
      jnp.where(use_sf, log_scaled_sf, 0.0), accuracy=AccuracyMode.HIGHEST
    ),
    0.5,
  )
  logcdf_open = jnp.where(
    use_sf, lax.log1p(-sf_safe, accuracy=AccuracyMode.HIGHEST), logcdf_in
  )
  out = jnp.where(z <= a, -np.inf, jnp.where(z >= b, 0.0, logcdf_open))
  out = jnp.where(valid_params & ~jnp.isnan(z), out, np.nan)
  return out.astype(orig_dtype)


@api.jit
def cdf(
  x: ArrayLike,
  a: ArrayLike,
  b: ArrayLike,
  loc: ArrayLike = 0,
  scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf} = \int_{-\infty}^x f_{pdf}(y) \mathrm{d}y

  where here :math:`f_{pdf}` is the probability distribution function,
  :func:`jax.scipy.stats.truncnorm.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  return lax.exp(logcdf(x, a, b, loc, scale))


@api.jit
def ppf(
  q: ArrayLike,
  a: ArrayLike,
  b: ArrayLike,
  loc: ArrayLike = 0,
  scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal percent point function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``ppf``.

  The percent point function is defined as the inverse of the cumulative
  distribution function, :func:`jax.scipy.stats.truncnorm.cdf`.

  Args:
    q: arraylike, value at which to evaluate the PPF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of ppf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.isf`
  """
  (q, a, b, loc, scale), orig_dtype = _promote_and_broadcast(
    "truncnorm.ppf", q, a, b, loc, scale
  )
  valid_params, a_safe, b_safe, _ = _sanitize_support(a, b, loc, scale)
  in_open = valid_params & (q > 0) & (q < 1)
  q_safe = jnp.where(in_open, q, 0.5)

  # Global CDF inversion: compute `log Phi(z) = logaddexp(log_ndtr(a),
  # log(q * mass))` so `ndtri_exp` can invert it. Use whichever half
  # (left or right tail) gives the smaller CDF value for better precision.
  x0_ab, w_ab, log_scaled_ab, quad_integral, use_quad = (
    _scaled_gauss_mass_with_quad(a_safe, b_safe)
  )
  log_scaled_total = (
    lax.log(w_ab, accuracy=AccuracyMode.HIGHEST) + log_scaled_ab
  )
  log_mass = _half_sq_diff(0.0, x0_ab) + log_scaled_total
  log_q = lax.log(q_safe, accuracy=AccuracyMode.HIGHEST)
  log_1mq = lax.log1p(-q_safe, accuracy=AccuracyMode.HIGHEST)
  log_q_mass = log_q + log_mass
  log_1mq_mass = log_1mq + log_mass

  # Skip `_logaddexp` when `a = -inf` or `b = +inf`: `log_ndtr'(-inf)` is
  # `+inf` and would produce `NaN` in the JVP.
  a_is_neginf = jnp.isneginf(a_safe)
  b_is_posinf = jnp.isposinf(b_safe)
  a_fin = jnp.where(a_is_neginf, 0.0, a_safe)
  b_fin = jnp.where(b_is_posinf, 0.0, b_safe)
  log_phi_z = jnp.where(
    a_is_neginf, log_q_mass, _logaddexp(log_ndtr(a_fin), log_q_mass)
  )
  log_phi_neg_z = jnp.where(
    b_is_posinf, log_1mq_mass, _logaddexp(log_ndtr(-b_fin), log_1mq_mass)
  )
  # Use `jnp.where` instead of `jnp.minimum` to avoid halving gradients at
  # the tie point. When `log_mass` underflows to `-inf` (extreme tails),
  # all interior quantiles collapse onto `x0_ab`.
  is_huge = jnp.isneginf(log_mass)
  use_left = log_phi_z <= log_phi_neg_z
  log_phi_left = jnp.where(use_left & ~is_huge, log_phi_z, -_LOG_2)
  log_phi_right = jnp.where(~use_left & ~is_huge, log_phi_neg_z, -_LOG_2)
  z_global = jnp.where(
    is_huge,
    x0_ab,
    jnp.where(use_left, ndtri_exp(log_phi_left), -ndtri_exp(log_phi_right)),
  )

  # Local refinement: when the quantile is near a finite endpoint,
  # global inversion loses the small step in rounding; solve the
  # near-endpoint integral equation directly.
  from_right = q_safe > 0.5
  a0 = jnp.where(from_right, -b_fin, a_fin)  # nearest endpoint
  a0_is_fin = jnp.where(from_right, ~b_is_posinf, ~a_is_neginf)
  p = jnp.where(from_right, 1.0 - q_safe, q_safe)
  log_p = jnp.where(from_right, log_1mq, log_q)
  x0_near = jnp.where(
    a0_is_fin, jnp.where(from_right, -x0_ab, x0_ab), 0.0
  )
  log_r0 = (
    log_p
    + log_scaled_total
    + _HALF_LOG_2PI
    + _half_sq_diff(a0, x0_near)
  )
  small_step = a0_is_fin & (log_r0 <= _LOG_LOCAL_STEP_MAX)
  # When using quadrature, shift from midpoint-centered to endpoint-
  # centered to avoid midpoint rounding error.
  h_quad = jnp.where(use_quad, 0.5 * w_ab, 0.0)
  r0 = jnp.where(
    use_quad,
    p
    * quad_integral
    * lax.exp(-h_quad * (a0 + 0.5 * h_quad), accuracy=AccuracyMode.HIGHEST),
    lax.exp(jnp.where(small_step, log_r0, 0.0), accuracy=AccuracyMode.HIGHEST),
  )
  use_local = small_step & (jnp.abs(a0 * r0) <= _LOCAL_STEP_MAX)
  delta = _ppf_local_step(*_mask_pair(use_local, a0, r0, 0.0, 0.0))
  z_local = jnp.where(from_right, -(a0 + delta), a0 + delta)

  z = _clamp_preserve_grad(
    jnp.where(use_local, z_local, z_global), a_safe, b_safe
  )
  # Handle `q == 0` and `q == 1` before scaling so `a = -inf` or
  # `b = +inf` doesn't produce `0 * inf = NaN` in `d(ppf)/d(scale)`.
  z = jnp.where(q == 0, a_safe, jnp.where(q == 1, b_safe, z))
  valid_q = (q >= 0) & (q <= 1) & ~jnp.isnan(q)
  out = jnp.where(valid_params & valid_q, loc + z * scale, np.nan)
  return out.astype(orig_dtype)


@api.jit
def isf(
  q: ArrayLike,
  a: ArrayLike,
  b: ArrayLike,
  loc: ArrayLike = 0,
  scale: ArrayLike = 1,
) -> Array:
  r"""Truncated normal inverse survival function.

  JAX implementation of :obj:`scipy.stats.truncnorm` ``isf``.

  Returns the inverse of the survival function,
  :func:`jax.scipy.stats.truncnorm.sf`.

  Args:
    q: arraylike, value at which to evaluate the ISF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of isf values.

  See Also:
    - :func:`jax.scipy.stats.truncnorm.cdf`
    - :func:`jax.scipy.stats.truncnorm.pdf`
    - :func:`jax.scipy.stats.truncnorm.sf`
    - :func:`jax.scipy.stats.truncnorm.logcdf`
    - :func:`jax.scipy.stats.truncnorm.logpdf`
    - :func:`jax.scipy.stats.truncnorm.logsf`
    - :func:`jax.scipy.stats.truncnorm.ppf`
  """
  q, a, b, loc, scale = promote_args_inexact(
    "truncnorm.isf", q, a, b, loc, scale
  )
  return -ppf(q, -b, -a, -loc, scale)
