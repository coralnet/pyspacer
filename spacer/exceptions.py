class ConfigError(Exception):
    pass


class DataLimitError(Exception):
    pass


class HashMismatchError(Exception):
    pass


class RowColumnInvalidError(Exception):
    pass


class RowColumnMismatchError(Exception):
    pass


class TrainingLabelsError(Exception):
    pass


class URLDownloadError(Exception):
    """
    This wraps around several different errors that can happen
    when trying to download from a URLStorage url. This helps to
    indicate that the exceptions are from a URLStorage url and not
    some other kind of download/read.
    The original error's saved in a field in case the handler wants
    to handle different error causes in different ways.
    """
    def __init__(self, message, original_error):
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self):
        return (
            f"{self.message} /"
            f" Details - {type(self.original_error).__name__}:"
            f" {self.original_error}"
        )
