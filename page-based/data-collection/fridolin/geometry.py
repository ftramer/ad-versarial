import itertools

#
# Credit: Oleh Prypin, https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles/25068722#25068722
#

class Rectangle:
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            return Rectangle(x1, y1, x2, y2)
    __and__ = intersection
    
    def difference(self, other):
        inter = self & other
        if not inter:
            yield self
            return
        xs = {self.x1, self.x2}
        ys = {self.y1, self.y2}
        if self.x1 < other.x1 < self.x2: xs.add(other.x1)
        if self.x1 < other.x2 < self.x2: xs.add(other.x2)
        if self.y1 < other.y1 < self.y2: ys.add(other.y1)
        if self.y1 < other.y2 < self.y2: ys.add(other.y2)
        for (x1, x2), (y1, y2) in itertools.product(
            pairwise(sorted(xs)), pairwise(sorted(ys))
        ):
            rect = type(self)(x1, y1, x2, y2)
            if rect != inter:
                yield rect
    __sub__ = difference
    
    def width(self):
        return self.x1 + self.x2

    def height(self):
        return self.y1 + self.y2

    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
    
    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2
    
    def __eq__(self, other):
        return isinstance(other, Rectangle) and tuple(self) == tuple(other)
    
    def __ne__(self, other):
        return not (self == other)
    
    def __repr__(self):
        return "Rect" + repr(tuple(self))


def pairwise(iterable):
    # //docs.python.org/dev/library/itertools.html#recipes
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)