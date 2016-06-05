import data_sets


def test():
    ds = data_sets.Data_set([1, 2, 3], ['a', 'b', 'c'])
    print (ds.next_batch(1))
    print (ds.next_batch(2))
    print (ds.next_batch(3))
    print (ds.next_batch(1))
    assert False
