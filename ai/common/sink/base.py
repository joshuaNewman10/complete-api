from abc import abstractmethod


class Sink:
    """
    Base sink class
    """
    def filter(self, _):
        return True

    def transform(self, item):
        return item

    @abstractmethod
    def sink(self, item, **kwargs):
        raise NotImplementedError

    def receive(self, item, **kwargs):
        if self.filter(item):
            item = self.transform(item)
            return self.sink(item, **kwargs)
