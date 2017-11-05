import sys
from stats import Stats
from bin import Bin

'''
Program entry of statistical analysis and binning

Author: Shaobo Wang

'''


class StatsBin:
    options = ''

    # constructor
    def __init__(self, options):
        self.options = options

    # main method
    def main(self):
        st = Stats(self.options)
        st.get_stats()
        bn = Bin(self.options)
        bn.bin()


if __name__ == '__main__':
    stats_bin = StatsBin(sys.argv)
    stats_bin.main()
