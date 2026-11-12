"""NKI tensor views shared by the plugin's kernels."""


def broadcast_column(tile, size: int):
    """View a ``[P, 1]`` SBUF tile as ``[P, size]`` repeating its single column.

    The access pattern nkilib's ``TensorView(tile).broadcast(dim=1, size=size)
    .get_view()`` emits for such a tile (row-major strides, stride 0 on the
    broadcast dim); an nkilib release without ``core.utils.tensor_view`` still
    has the underlying ``.ap``.
    """
    return tile.ap(pattern=[(1, tile.shape[0]), (0, size)], offset=0, dtype=tile.dtype)
