def __getattr__(name):
    if name == "CalibrationMap":
        from .calibration import CalibrationMap
        return CalibrationMap
    if name == "ZoneMap":
        from .zone_map import ZoneMap
        return ZoneMap
    raise AttributeError(f"module 'spatial' has no attribute {name!r}")

__all__ = ["CalibrationMap", "ZoneMap"]
