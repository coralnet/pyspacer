import abc

from spacer.data_classes import DataLocation


class Storage(abc.ABC):  # pragma: no cover

    @abc.abstractmethod
    def load(self, loc: DataLocation) -> BytesIO:
        pass

