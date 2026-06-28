"""training subpackage."""

from .losses import CategoryLossConfig, category_loss

__all__ = ["CategoryLossConfig", "category_loss", "preprocess_dataset"]


def __getattr__(name: str):
    # 惰性导出：避免 `python -m frame_1.training.preprocess` 的 re-import 警告，
    # 也避免包导入时即加载 preprocess 的重依赖。
    if name == "preprocess_dataset":
        from .preprocess import preprocess_dataset
        return preprocess_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
