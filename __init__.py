from .PainterSamplerLTXV import PainterSamplerLTXV
from .PainterLTX2VPlus import PainterLTX2VPlus
from .PainterLTX2V import PainterLTX2V

NODE_CLASS_MAPPINGS = {
    "PainterSamplerLTXV": PainterSamplerLTXV,
    "PainterLTX2VPlus": PainterLTX2VPlus,
    "PainterLTX2V": PainterLTX2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterSamplerLTXV": "Painter Sampler LTXV",
    "PainterLTX2VPlus": "Painter LTX2V Plus",
    "PainterLTX2V": "Painter LTX2V",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
