
from sklearn.datasets import make_blobs


x = (3, 0, 3)

match x:
    case (int(a), int(b), int(c)):
        print("3 ints", a, b, c)
    case _:
        print("other")
