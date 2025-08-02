from modules import models


def list_models():
    models.list_models()

    return list(models.models)
