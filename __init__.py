"""AEX Drift: hold an obvpm Timeline extend to the tone of the clip it continues, for MiniMax H3."""

from .nodes import DriftExtension

WEB_DIRECTORY = "./web"


async def comfy_entrypoint():
    return DriftExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
