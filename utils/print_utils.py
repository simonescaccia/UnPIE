num_dashes = 100

def print_separator(message=None, space=True, top_new_line=True, bottom_new_line=True):
    if space and top_new_line:
        print()
    print('-'*num_dashes)
    if message:
        print(message)
    if space and bottom_new_line:
        print()
    return