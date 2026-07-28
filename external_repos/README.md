# Minimal external references

This directory does not vendor complete external repositories. It contains only
two small files required by the current scientific tests and FK implementation:

```text
kimodo/kimodo/assets/skeletons/smplx22/joints.p
  SMPL-X22 neutral rest joints used by HY273 FK losses and metrics.
  Source: https://github.com/nv-tlabs/kimodo
  SHA256: cb4a2fdb4f1a5b49314b3e13eb33bf35700f4a2aef09c7a1976123f9154976e9

273_motion_raw_diffusion_plan/models/codeflow/dit_blocks.py
  Pre-fusion-change reference used to test exact f00 compatibility.
  SHA256: 564cec85590095a84b2c8e0b2f6b738412c8d103d2f2b40fab380273a05ec904
```

The Kimodo Apache-2.0 license is included next to the copied rest-skeleton
asset. Full external repositories remain local dependencies for some optional
benchmarks.
