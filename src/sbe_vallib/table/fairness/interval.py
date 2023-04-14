import pandas as pd


class Interval(pd.Interval):
    def __init__(self, left, right, closed='neither'):
        super().__init__(left, right, closed)

    def __and__(self, other: pd.Interval):
        if not self.overlaps(other):
            return pd.Interval(0, 0, closed='neither')  # empty set
        left = max(self.left, other.left)
        right = min(self.right, other.right)
        left_in = (left in self) and (left in other)
        right_in = (right in self) and (right in other)
        closed = 'neither'
        if left_in and right_in:
            closed = 'both'
        if left_in and not right_in:
            closed = 'left'
        if not left_in and right_in:
            closed = 'right'
        return pd.Interval(left, right, closed)
