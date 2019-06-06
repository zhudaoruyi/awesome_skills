import os
import sys
from os.path import join


def main():
    file_names = os.listdir(sys.argv[1])
    for fn in file_names:
        os.rename(join(sys.argv[1], fn), join(sys.argv[1], sys.argv[1]+"_"+fn))
    return


if __name__ == '__main__':
    main()
