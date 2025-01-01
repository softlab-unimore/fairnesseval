import argparse
import logging
import os
import sys
import warnings


class DeprecateAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        warnings.warn("Argument %s is deprecated and is *ignored*." % self.option_strings)
        # delattr(namespace, self.dest)

def intersection_sorted(a,b):
    # Intersection of two lists keeping the order of the first list
    a = list(a)
    return sorted(list(set(a) & set(b)), key = lambda x: a.index(x))

def difference_sorted(a,b):
    # Difference of two lists keeping the order of the first list
    a = list(a)
    return sorted(list(set(a) - set(b)), key = lambda x: a.index(x))

def mark_deprecated_help_strings(parser, prefix="DEPRECATED"):
    for action in parser._actions:
        if isinstance(action, DeprecateAction):
            h = action.help
            if h is None:
                action.help = prefix
            else:
                action.help = prefix + ": " + h


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_project_root() -> str:
    tpath = os.path.join(os.path.dirname(__file__), '..', '..')
    return os.path.abspath(tpath)



def init_logger(save_dir: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    # reset the logger
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)
    #     handler.close()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(save_dir, 'run.log'), mode='a')
    # c_handler.setLevel(logging.INFO)
    # f_handler.setLevel(logging.INFO)
    format = logging.Formatter(fmt='%(asctime)s %(levelname)s:%(name)s: %(message)s',
                                 datefmt='%d/%m/%y %H:%M:%S', )
    c_handler.setFormatter(format)
    f_handler.setFormatter(format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    def except_hook(exc_type, exc_value, exc_traceback):
        # Handle exception
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default KeyboardInterrupt handler
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        # Then propagate the exception
        raise exc_value

    sys.excepthook = except_hook

    # def exc_handler(exctype, value, tb):
    #     logging.exception(''.join(traceback.format_exception(exctype, value, tb)))
    #
    # sys.excepthook = exc_handler

    return logger

# class logger singleton pattern
class LoggerSingleton:
    _instance = None

    def __new__(cls, reset=False, *args, **kwargs):
        if not cls._instance or reset:
            cls._instance = init_logger(*args, **kwargs)
        return cls._instance

    @staticmethod
    def close(_instance):
        for handler in _instance.handlers[:]:
            _instance.removeHandler(handler)
            handler.close()
            del handler
        del _instance