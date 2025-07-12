class BaseConverter:
    """Custom base conversion utility."""
    def __init__(self, digits):
        self.digits = digits
        self.base = len(digits)

    def encode(self, number):
        if number == 0:
            return self.digits[0]
        result = []
        while number:
            number, remainder = divmod(number, self.base)
            result.append(self.digits[remainder])
        return ''.join(reversed(result))

    def decode(self, string):
        number = 0
        for char in string:
            number = number * self.base + self.digits.index(char)
        return number


# Example converters
base62 = BaseConverter('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
base16 = BaseConverter('0123456789ABCDEF')
