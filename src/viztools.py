from __future__ import print_function


def test(key_list):
    for key, val in enumerate(key_list):
        if val:
            if key != 255:
                print(key)
            key_list[key] = False