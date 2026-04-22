from __future__ import annotations
from typing import Callable

class LazyDataset[T]:
    def __init__(self, n: int, f: Callable[[int], T]) -> None:
        self.n = n
        self.f = f
    
    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.f(i)
    
    def map[V](self, g: Callable[[T], V]) -> LazyDataset[V]:
        return LazyDataset[V](
            n=self.n,
            f=lambda i: g(self.f(i)),
        )
