from .lift_3d import Lift3DNode

NODE_CLASS_MAPPINGS = {
    "Lift3D": Lift3DNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lift3D": "Lift 3D (Pose + Depth → Unity)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
