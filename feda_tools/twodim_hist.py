def hello_world():
    print("This is my first pip package!")

def cmd_args(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parsed_args = parser.parse_args(args)

    print((parsed_args.x, parsed_args.y))
