def model_type_definer(model_type: str, baseline: bool, signature: bool):
    if signature and baseline:
        raise ValueError("Cannot have both signature and baseline enabled.")
    prefix = "_baseline" if baseline else "_signature" if signature else ""
    return  f"{model_type}{prefix}"



