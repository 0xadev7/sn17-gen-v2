import io
import trimesh as tm


def mesh_to_binary_ply_bytes(mesh) -> bytes:
    """
    Export a trimesh.Trimesh to binary little-endian PLY bytes.
    """
    buf = io.BytesIO()
    if hasattr(mesh, "export"):
        # trimesh path
        mesh.export(buf, file_type="ply", encoding="binary_little_endian")
        return buf.getvalue()
    # If your mesh object isn't a trimesh, adapt this block accordingly.
    raise RuntimeError(
        "mesh_to_binary_ply_bytes: mesh type not supported by fallback exporter"
    )
