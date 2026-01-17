class ListNode:
    @classmethod
    def to_list(cls, head: "ListNode | None") -> list[int]:
        """
        Convert a linked list to a python list

        Args:
            head (ListNode | None): The first node in the list

        Returns:
            result (list[int]): The converted list of integers
        """
        if not head:
            return []

        result: list[int] = []
        while head:
            result.append(head.value)
            head = head.next

        return result

    @classmethod
    def from_list(cls, values: list[int]) -> "tuple[ListNode | None, ListNode | None]":
        """
        Convert a python list to a linked list

        Args:
            values (list[int]): Python list to convert

        Returns:
            (head, tail) tuple(ListNode | None, ListNode | None): The head and tail node of the linked list
        """
        if len(values) == 0:
            return (None, None)
        elif len(values) == 1:
            node: ListNode = ListNode(values[0])
            return (node, node)

        head: ListNode = ListNode(values[0])
        tail: ListNode = head

        for v in values[1:]:
            tail.next = ListNode(v)
            tail = tail.next

        return head, tail

    def __init__(self, value: int, next: "ListNode | None" = None):
        """
        Initialze the linked list node with the given value and optional next node

        Args:
            value (int): Value of the current node
            next (ListNode | None): Optional next node of the current node
        """
        self.value: int = value
        self.next: "ListNode | None" = next
