from nora.config import locate

__all__ = ["build_model"]


def build_model(cfg):
    cls = cfg.pop("_target_", None)

    if isinstance(cfg, str):
        cls_name = cls
        cls = locate(cls_name)
        assert cls is not None, cls_name

    return cls(**cfg)
