import io
import trimesh as tm


def mesh_to_binary_ply_bytes(mesh: tm.Trimesh) -> bytes:
    # Make sure normals/colors exist if you want them in the PLY
    _ = mesh.vertex_normals  # triggers compute if missing
    if mesh.visual is None or mesh.visual.vertex_colors is None:
        # optional: add a default gray color
        mesh.visual.vertex_colors = [200, 200, 200, 255]

    buf = io.BytesIO()
    # Binary PLY (little endian)
    mesh.export(buf, file_type="ply", encoding="binary_little_endian")
    return buf.getvalue()
