def tune_prompt(base_prompt: str) -> str:
    is_pink_object = any(
        color in base_prompt.lower() for color in ["pink", "magenta", "rose", "fuchsia"]
    )

    return (
        f"{base_prompt}, studio product photo, centered, full body in frame, front-left three-quarter view, "
        f"eye-level camera, on seamless {'pink' if not is_pink_object else 'grey'} background, soft even lighting, "
        f"subtle contact shadow on ground, sharp focus, high detail, photorealistic, 8k"
    )


negative_prompt = (
    f"cropped, out of frame, occluded, multiple objects, busy background, "
    f"text, watermark, logo, lens flare, deep shadows, motion blur, "
    f"extreme bokeh, isometric, orthographic, fisheye "
)
