from heapq import heappush, heappop
from typing import Optional, List, Iterable, Protocol

from events import Events

from .node import Node


class NodeQueue(Protocol):
    def __init__(self, event_listener: Events, *args, **kwargs):
        pass

    def __len__(self):
        pass

    def has_next_node(self) -> bool:
        pass

    def enqueue(self, node: Node):
        pass

    def peek(self) -> Optional[Node]:
        pass

    def pop(self) -> Node:
        pass

    def __iter__(self) -> Iterable[Node]:
        pass


class FIFONodeQueue:
    def __init__(self, *args, **kwargs):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def has_next_node(self) -> bool:
        return len(self.queue) > 0

    def enqueue(self, node: Node):
        self.queue.append(node)

    def peek(self) -> Optional[Node]:
        return self.queue[0] if len(self.queue) > 0 else None

    def pop(self) -> Node:
        if len(self.queue) == 0:
            raise ValueError
        return self.queue.pop(0)

    def __iter__(self) -> Iterable[Node]:
        return iter(self.queue)


class LIFONodeQueue:
    def __init__(self, *args, **kwargs):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def has_next_node(self) -> bool:
        return len(self.queue) > 0

    def enqueue(self, node: Node):
        self.queue.append(node)

    def peek(self) -> Optional[Node]:
        return self.queue[-1] if len(self.queue) > 0 else None

    def pop(self) -> Node:
        if len(self.queue) == 0:
            raise ValueError
        return self.queue.pop()

    def __iter__(self) -> Iterable[Node]:
        return iter(self.queue)


class LowerBoundNodeWrapper:
    def __init__(self, node: Node):
        self.node = node

    def __lt__(self, other):
        return self.node.lower_bound < other.node.lower_bound


class BestBoundQueue:
    def __init__(self, *args, **kwargs):
        self._node_queue: List[LowerBoundNodeWrapper] = []

    def __len__(self):
        return len(self._node_queue)

    def has_next_node(self) -> bool:
        return len(self._node_queue) > 0

    def enqueue(self, node: Node):
        heappush(self._node_queue, LowerBoundNodeWrapper(node))

    def peek(self) -> Optional[Node]:
        return self._node_queue[0].node if self.has_next_node() else None

    def pop(self) -> Node:
        return heappop(self._node_queue).node

    def __iter__(self) -> Iterable[Node]:
        return (x.node for x in self._node_queue)


class TwoStageQueue:
    def __init__(self, event_handler, *args, **kwargs):
        self.queue = LIFONodeQueue(event_handler, *args, **kwargs)
        self.event_handler = event_handler
        self.event_handler.on_integral_solution_found += lambda n: self._on_integral_solution_found(*args, **kwargs)

    def _on_integral_solution_found(self, *args, **kwargs):
        if isinstance(self.queue, LIFONodeQueue):
            self._switch_queue_impl(BestBoundQueue(self.event_handler, *args, **kwargs))

    def has_next_node(self) -> bool:
        return self.queue.has_next_node()

    def __len__(self):
        return len(self.queue)

    def _switch_queue_impl(self, queue: NodeQueue):
        while self.queue.has_next_node():
            queue.enqueue(self.queue.pop())
        self.queue = queue

    def enqueue(self, node: Node):
        self.queue.enqueue(node)

    def peek(self) -> Optional[Node]:
        return self.queue.peek()

    def pop(self) -> Node:
        return self.queue.pop()

    def __iter__(self) -> Iterable[Node]:
        return iter(self.queue)
