import sys
from histogram import Histogram
from correlation import Correlation

'''
Program entry of histogram and correlation

Author: Shaobo Wang

'''


class HistCor:
    options = ''

    # constructor
    def __init__(self, options):
        self.options = options

    # main method
    def main(self):
        # hg = Histogram(self.options)
        # hg.hist()
        cr = Correlation(self.options)
        cr.corr()


if __name__ == '__main__':
    hist_cor = HistCor(sys.argv)
    hist_cor.main()
