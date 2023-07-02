import numpy as np

def simhash(vectors, hashkeys):
    bools = vectors.dot(hashkeys) > 0
    hashvec = np.sum(np.array([bools[:, i] << i for i in range(hashkeys.shape[1])]), axis=0)
    return hashvec


if __name__ == '__main__':
    len_vec = 1024
    len_hash = 3 # creates 2**len_hash partitions of space
    num_rows = 8

    hashkeys = np.random.randn(len_vec, len_hash)
    vectors = np.random.randn(num_rows, len_vec)
    
    print(simhash(vectors, hashkeys))
    print([np.binary_repr(h, width=len_hash) for h in simhash(vectors, hashkeys)])
    
    # saving hash keys
    # np.save('hashkeys.npy', hashkeys)
    # loading hash keys
    # hashkeys = np.load('hashkeys.npy')
