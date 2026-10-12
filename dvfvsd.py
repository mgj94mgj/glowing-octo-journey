import string
import random


def generate_random_password():
    """生成 6 位随机密钥，每位为 ASCII 字母或数字。"""
    chars = string.ascii_letters + string.digits
    return [ord(random.choice(chars)) for _ in range(6)]


if __name__ == "__main__":
    current_password = generate_random_password()
    print(current_password)
    password_text = "".join(chr(code) for code in current_password)
    print(password_text)
    print([ord(char) for char in reversed(password_text.strip())])
