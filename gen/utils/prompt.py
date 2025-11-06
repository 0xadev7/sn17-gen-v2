def tune_prompt(base_prompt: str) -> str:
    return (
        f"{base_prompt}, studio product photo, centered, full body in frame, front-left three-quarter view, "
        f"eye-level camera, on seamless {'grey' if 'grey' in base_prompt else 'neutral'} background, soft even lighting, "
        "subtle contact shadow on ground, sharp focus, high detail, photorealistic, 8k"
    )
