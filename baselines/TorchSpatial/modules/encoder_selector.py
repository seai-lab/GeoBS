import importlib
from typing import Any, Dict, Tuple, Union, Type

# The entries' names are copied from the documentation.
ENCODER_REGISTRY: Dict[str, Union[type, Tuple[str, str]]] = {
    # name: (module_path, class_name)
    "Space2Vec-grid": ("..location_encoders.GridCellSpatialRelationLocationEncoder", "GridCellSpatialRelationLocationEncoder"),
    "Space2Vec-theory": ("..location_encoders.TheoryGridCellSpatialRelationLocationEncoder", "TheoryGridCellSpatialRelationLocationEncoder"),
    "xyz": ("..location_encoders.XYZSpatialRelationLocationEncoder", "XYZSpatialRelationLocationEncoder"),
    "NeRF": ("..location_encoders.NERFSpatialRelationLocationEncoder", "NERFSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereC": ("..location_encoders.SphereSpatialRelationLocationEncoder", "SphereSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereC+": ("..location_encoders.SphereGridSpatialRelationLocationEncoder", "SphereGridSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereM": ("..location_encoders.SphereMixScaleSpatialRelationLocationEncoder", "SphereMixScaleSpatialRelationLocationEncoder"),
    "Sphere2Vec-sphereM+": ("..location_encoders.SphereGridMixScaleSpatialRelationLocationEncoder", "SphereGridMixScaleSpatialRelationLocationEncoder"),
    "Sphere2Vec-dfs": ("..location_encoders.DFTSpatialRelationLocationEncoder", "DFTSpatialRelationLocationEncoder"),
    "rbf": ("..location_encoders.RBFSpatialRelationLocationEncoder", "RBFSpatialRelationLocationEncoder"),
    "rff": ("..location_encoders.RFFSpatialRelationLocationEncoder", "RFFSpatialRelationLocationEncoder"),
    "wrap": ("..modules.models", "FCNet"),
    "wrap_ffn": ("..location_encoders.AodhaFFNSpatialRelationLocationEncoder", "AodhaFFNSpatialRelationLocationEncoder"),
    "tile_ffn": ("..location_encoders.GridLookupSpatialRelationLocationEncoder", "GridLookupSpatialRelationLocationEncoder"),
    "Siren(SH)": ("..location_encoders.SphericalHarmonicsSpatialRelationLocationEncoder", "SphericalHarmonicsSpatialRelationLocationEncoder"),
    # These below are not in the allowed options in tutorial.ipynb but Nemin says they can be implemented
    # Not supported by the current training implementation
    "GridCellNorm": ("..location_encoders.GridCellNormSpatialRelationEncoder", "GridCellNormSpatialRelationEncoder"),
    "HexagonGridCell": ("..location_encoders.HexagonGridCellSpatialRelationEncoder", "HexagonGridCellSpatialRelationEncoder"),
    "Naive": ("..location_encoders.NaiveSpatialRelationEncoder", "NaiveSpatialRelationEncoder"),
    "TheoryDiagGridCellSpatialRelationEncoder": ("..location_encoders.TheoryDiagGridCellSpatialRelationEncoder", "TheoryDiagGridCellSpatialRelationEncoder"),
    "": ("..location_encoders.", ""),
}

def _resolve_encoder(name: str) -> type:
    entry = ENCODER_REGISTRY[name]
    if isinstance(entry, type):
        return entry  # already resolved / cached

    module_path, class_name = entry
    mod = importlib.import_module(module_path, package="TorchSpatial.modules")
    cls = getattr(mod, class_name) # cls means the actual class object

    ENCODER_REGISTRY[name] = cls  # cache for next time
    return cls

def get_loc_encoder(name: str, overrides: Dict[str, Any] | None = None, **kwargs):
    EncoderCls = _resolve_encoder(name)
    cfg = {**kwargs, **(overrides or {})} # **kwargs implies no need to hard code encoder-specific parameters
    return EncoderCls(**cfg)

