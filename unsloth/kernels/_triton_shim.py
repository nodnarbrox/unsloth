# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fake triton module for Maxwell GPUs (compute capability < 7.0).

Provides no-op stubs for triton.jit, triton.heuristics, and triton.language
so that kernel source files can be parsed without ImportError. The decorated
functions are never called on Maxwell GPUs -- the public API functions are
overridden with PyTorch fallbacks.
"""

from ..device_type import IS_MAXWELL_GPU


def _noop_decorator(fn):
    """No-op decorator that returns the function unchanged."""
    return fn


def _noop_heuristics(mapping):
    """No-op heuristics that returns a no-op decorator."""
    return _noop_decorator


def _noop_cdiv(a, b):
    return (a + b - 1) // b


def _noop_next_power_of_2(x):
    n = 1
    while n < x:
        n *= 2
    return n


class _FakeConstexpr:
    """Placeholder for tl.constexpr type annotations."""
    pass


class _FakeLanguage:
    """Fake triton.language module."""
    constexpr = _FakeConstexpr
    int32 = None
    int64 = None
    float32 = None


class _FakeTriton:
    """Fake triton module with no-op jit and heuristics."""
    jit = staticmethod(_noop_decorator)
    heuristics = staticmethod(_noop_heuristics)
    cdiv = staticmethod(_noop_cdiv)
    next_power_of_2 = staticmethod(_noop_next_power_of_2)
    language = _FakeLanguage()


# When IS_MAXWELL_GPU is True, callers do:
#   from ._triton_shim import triton, tl
# instead of:
#   import triton
#   import triton.language as tl

if IS_MAXWELL_GPU:
    triton = _FakeTriton()
    tl = triton.language
else:
    import triton as triton
    import triton.language as tl
