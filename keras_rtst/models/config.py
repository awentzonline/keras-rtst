import importlib


MODEL_CONFIG = {
    'transfer': {
        'input_generator': 'keras_rtst.generators.style_xfer',
        'evaluation_input_generator': 'keras_rtst.generators.style_xfer',
        'make_model': 'keras_rtst.models.style_xfer'
    },
    'girthy': {
        'input_generator': 'keras_rtst.generators.style_xfer',
        'evaluation_input_generator': 'keras_rtst.generators.style_xfer',
        'make_model': 'keras_rtst.models.style_xfer_girthy'
    }
}


def get_model_by_name(name):
    conf = MODEL_CONFIG[name]
    results = []
    output_order = ('make_model', 'input_generator', 'evaluation_input_generator')
    for prop in output_order:
        value = getattr(importlib.import_module(conf[prop]), prop)
        results.append(value)
    return results
