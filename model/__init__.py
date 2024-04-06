def load_model(model_name, *args):
    # Choose model
    if model_name == "normal":
        from model.normal18 import ResNet18
    elif model_name == "modified":
        from model.modified18 import ResNet18
    else:
        raise ValueError("Invalid model_type. Choose 'normal' or 'modified'.")
    return ResNet18(*args)