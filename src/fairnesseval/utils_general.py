import argparse
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
