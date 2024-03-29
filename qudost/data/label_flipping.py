# label_flipping.py

def flip_parity_label(label):
    return int(label) % 2  # 0 if label is even, 1 if label is odd

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def flip_primality_label(label):
    return int(is_prime(label))

def flip_loop_label(label):
    return int(label in [0, 4, 6, 8, 9])

def flip_mod_3_label(label):
    return int(label) % 3

def flip_mod_4_label(label):
    return int(label) % 4

def flip_mod_3_binary_label(label):
    return int(label > 0)  # 1 if label is not divisible by 3, 0 otherwise

def flip_mod_4_binary_label(label):
    return int(label > 0)  # 1 if label is not divisible by 4, 0 otherwise

def flip_0_to_4_binary_label(label):
    return int(label < 5)  # 1 if label is between 0-4 (inclusive), 0 otherwise

def plus_1(label):
    return int((label + 1) % 10)

def squash_3(label):
    return int(label//3)

def squash_4(label):
    return int(label//4)

def find_0(label):
    return(int(label in [0]))

def find_1(label):
    return(int(label in [1]))

def find_2(label):
    return(int(label in [2]))
def find_3(label):
    return(int(label in [3]))

def find_4(label):
    return(int(label in [4]))

def find_5(label):
    return(int(label in [5]))

def find_6(label):
    return(int(label in [6]))
def find_7(label):
    return(int(label in [7]))
def find_8(label):
    return(int(label in [8]))

def find_9(label):
    return(int(label in [9]))