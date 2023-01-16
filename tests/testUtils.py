import unittest

from utils import count_number_calls


class TestCountNumberCalls(unittest.TestCase):
    def setUp(self) -> None:
        @count_number_calls
        def arithmetic_sum(n):
            if n == 0:
                return 0
            return arithmetic_sum(n - 1) + n

        self.function = arithmetic_sum

    def test_correct_number_calls(self):
        for i in range(10):
            self.function(i)
            self.assertEqual(self.function.calls, i + 1)
            self.function.calls = 0

    def test_correct_output(self):
        for i in range(10):
            expected = i * (i + 1) // 2
            output = self.function(i)
            self.assertEqual(expected, output)


if __name__ == '__main__':
    unittest.main()
