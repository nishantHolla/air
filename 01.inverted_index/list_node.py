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
    def from_list(
        cls, values: list[int]
    ) -> "tuple[ListNode | None, ListNode | None, int]":
        """
        Convert a python list to a linked list

        Args:
            values (list[int]): Python list to convert

        Returns:
            (head, tail, length) tuple(ListNode | None, ListNode | None, int):
                The head and tail node of the linked list and its length
        """
        if len(values) == 0:
            return (None, None, 0)
        elif len(values) == 1:
            node: ListNode = ListNode(values[0])
            return (node, node, 1)

        head: ListNode = ListNode(values[0])
        tail: ListNode = head

        for v in values[1:]:
            tail.next = ListNode(v)
            tail = tail.next

        return head, tail, len(values)

    @classmethod
    def make_skips(cls, head: "ListNode | None", skip_n: int) -> None:
        """
        Initialize the skip pointers of nodes in a list to make it a skip list

        Args:
            head (ListNode | None): Head node of the list
            skip_n (int): Number of nodes to skip for each skip node
        """
        if not head or skip_n < 2:
            return

        fast: ListNode = head
        while fast.next:
            slow: ListNode = fast

            for _ in range(skip_n):
                if not fast.next:
                    break

                fast = fast.next

            slow.skip = fast

    def __init__(
        self, value: int, next: "ListNode | None" = None, skip: "ListNode | None" = None
    ):
        """
        Initialze the linked list node with the given value and optional next node

        Args:
            value (int): Value of the current node
            next (ListNode | None): Optional next node of the current node
            skip (ListNode | None): Optional skip node of the current node
        """
        self.value: int = value
        self.next: "ListNode | None" = next
        self.skip: "ListNode | None" = skip
