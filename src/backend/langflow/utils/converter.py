def copy_object(source, destination_cls):
    source_data = {attr: getattr(source, attr) for attr in dir(source) if not callable(getattr(source, attr)) and not attr.startswith("__")}
    return destination_cls(**source_data)