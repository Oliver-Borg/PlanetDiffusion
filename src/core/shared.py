
class ArgsKwargsWrapper:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
