class ConfigError(Exception):
    pass


class HashMismatchError(Exception):
    pass


class SpacerInputError(Exception):
    """
    This should indicate that the exception was not caused by a spacer bug, but
    by giving inputs that spacer cannot reasonably handle. For example, an
    unreachable URL given as the image URL to download.
    """
    pass
