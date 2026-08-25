"""SHASP model factory."""

from .shasp_model import ShaspModel


MODEL_CLASSES = {
    'shasp': ShaspModel,
}


def _model_class(model_name):
    try:
        return MODEL_CLASSES[model_name.lower()]
    except KeyError as exc:
        raise ValueError(
            'Supported models are {} (got {!r})'.format(
                ', '.join(sorted(MODEL_CLASSES)), model_name
            )
        ) from exc


def get_option_setter(model_name):
    return _model_class(model_name).modify_commandline_options


def create_model(opt):
    instance = _model_class(opt.model)(opt)
    print('model [{}] was created'.format(type(instance).__name__))
    return instance
