# pylint: disable=invalid-name,missing-docstring,exec-used,too-many-arguments,too-few-public-methods,no-self-use
from __future__ import print_function

import sys
from color import Color
from daisy import Daisy
from DB import Database
from edge import Edge
from evaluate import infer
from gabor import Gabor
from HOG import HOG
from resnet import ResNetFeat
from vggnet import VGGNetFeat

depth = 5
d_type = 'd1'
query_idx = 0

if __name__ == '__main__':
    db = Database()

    # methods to use
    methods = {
        "color": Color,
        "daisy": Daisy,
        "edge": Edge,
        "hog": HOG,
        "gabor": Gabor,
        "vgg": VGGNetFeat,
        "resnet": ResNetFeat
    }

    try:
        mthd = sys.argv[1].lower()
        query_idx = int(sys.argv[2].strip())
    except IndexError:
        print("usage: {} <method>".format(sys.argv[0]))
        print("supported methods:\ncolor, daisy, edge, gabor, hog, vgg, resnet")

        sys.exit(1)

    # call make_samples(db) accordingly
    samples = getattr(methods[mthd](), "make_samples")(db)

    # query the first img in data.csv
    query = samples[query_idx]
    print("\n[+] query: {}\n".format(query["img"]))

    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)

    for match in result:
        print("{}:\t{},\tClass {}".format(match["img"],
                                          match["dis"],
                                          match["cls"]))
