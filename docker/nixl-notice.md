# NIXL in this image

This image carries NIXL 1.5.0 built from source with the patches in this directory. It is not NIXL's PyPI wheel.

- **Source:** https://github.com/ai-dynamo/nixl, branch release/1.5.0 at commit
  075b2397029fc67a0b69a34df566c1ae3d00f5ab, the commit NIXL's 1.5.0 wheels on PyPI were built at.
- **Patches:** the files `0001`-`0009` beside this note, applied in order. The patched source is git tree
  6c6792fde6bdcc5d6da18de75141daca1b4764d6. `README.md` here says what each patch fixes.
- **Build:** sglang-neuron's `tools/build_nixl.sh`. It builds one wheel, `nixl-cu13` 1.5.0, holding libnixl, its Python
  bindings and only the LIBFABRIC transport plugin. The build has no etcd and no UCX. NIXL's meta package `nixl` 1.5.0
  is installed beside it, from PyPI.
- **Licenses:** NIXL's own license files are in the installed distributions' metadata:
  `site-packages/nixl_cu13-1.5.0.dist-info/licenses/` and `site-packages/nixl-1.5.0.dist-info/licenses/`.
  - NIXL's package metadata lists `licenses/NVIDIA-proprietary-LICENSE.txt` for every wheel built from its source.
  - Per NIXL's README, that license covers the NVIDIA UCX modules that NIXL's PyPI wheels bundle (`libuct_ib_mlx5_ext.so`,
    `libuct_ib_mlx5_gda.so`, `libuct_ib_mlx5_gdp.so`). This build has no UCX plugin and none of those modules.
