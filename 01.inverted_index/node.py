class Node:
    @classmethod
    def to_list(cls, head: "Node | None") -> list[int]:
        if not head:
            return []

        result = []
        while head:
            result.append(head.value)
            head = head.next

        return result

    @classmethod
    def from_list(cls, values: list[int]) -> "Node | None":
        if len(values) == 0:
            return None
        elif len(values) == 1:
            return Node(values[0])

        head = Node(values[0])
        tail = head

        for v in values[1:]:
            tail.next = Node(v)
            tail = tail.next

        return head

    @classmethod
    def get_tail(cls, node: "Node") -> "Node":
        while node.next:
            node = node.next

        return node

    def __init__(self, value: int, next: "Node | None" = None):
        self.value = value
        self.next = next
