import numpy as np


class Aux(object):
    def __init__(self):
        pass

    @staticmethod
    # alternativ: np.array([(float(i)-min(tx))/(max(tx)-min(tx)) for i in tx])
    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range


class MetricMeter:
    """
    Computes and stores simple statistics of some metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.min = float('inf')
        self.max = -self.min
        self.last_max = 0
        self.last_min = 0
        self.current = None
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if val > self.max:
            self.max = val
            self.last_max = 0
        else:
            self.last_max += 1
        if val < self.min:
            self.min = val
            self.last_min = 0
        else:
            self.last_min += 1
        self.current = val
        self.sum += val
        self.count += 1
        self.mean = self.sum / self.count


class MeterCollection:
    def __init__(self, *names):
        for name in names:
            if name.startswith('_') or name in ('meters', 'update', 'reset'):
                raise ValueError(f'Invalid name `{name}`')
        self.meters = {name: MetricMeter() for name in names}

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.meters[name].update(value)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def __getattr__(self, name):
        if name in self.meters:
            return self.meters[name]
        else:
            return getattr(super(), name)

    def __repr__(self):
        s = ['{name}={value:.4f}'.format(name=name, value=meter.mean)
             for name, meter in self.meters.items()]
        s = ' '.join(s)
        return s
