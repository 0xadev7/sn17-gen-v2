NORMAL_OBJ_TEMPLATE = (
    f"[OBJECT], studio product photo, diffuse material with slight specular highlights, "
    f"in a soft light tent with large softboxes left/right/top creating even illumination, "
    f"on seamless matte neutral gray background (uniform, no texture), soft contact shadow only "
    f", no environment reflections, no props, elevation 15°, centered, f/16, 70mm, "
    f"realistic but tame specular highlights, clean silhouette separation "
)

GLASS_OBJ_TEMPLATE = (
    f"[OBJECT], studio product photo, clear borosilicate glass (thin walls, crisp edges), "
    f"in a soft light tent with cross-polarized diffuse lighting, subtle dual rim lights outlining the silhouette, "
    f"on seamless matte charcoal background (uniform, no texture), soft contact shadow only, no caustics, "
    f"no internal clutter, no label, no liquid, no condensation, elevation 15°, centered, f/16, 70mm, "
    f"hyper-real, high dynamic range but controlled highlights, clean edge separation "
)

SILVER_OBJ_TEMPLATE = (
    f"[OBJECT], studio product photo, polished chrome with controlled reflections, "
    f"in a diffuse light tent, large softboxes left/right/top creating long soft gradients, "
    f"cross-polarized to reduce glare, on seamless matte pure white background (uniform, no texture), "
    f"soft contact shadow only, no environment reflections, no props, elevation 15°, centered, f/16, 70mm, "
    f"fine micro-scratches, realistic but tame specular highlights, clean silhouette separation "
)


def tune_prompt(base_prompt: str) -> str:
    if "glass" in base_prompt.lower():
        template = GLASS_OBJ_TEMPLATE
    elif "silver" in base_prompt.lower() or "chrome" in base_prompt.lower():
        template = SILVER_OBJ_TEMPLATE
    else:
        template = NORMAL_OBJ_TEMPLATE

    return base_prompt.replace("[OBJECT]", template)


negative_prompt = (
    f"cluttered background, hard shadows, horizon line, gradients, texture, fingerprints, "
    f"heavy glare, blown highlights, chromatic aberration, color cast, text, watermark, "
    f"hands, reflections of studio gear, environment reflections, smudges, dust, noise, "
    f"bokeh, motion blur, vignette"
)
