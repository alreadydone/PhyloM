
from imports import *


class DataSet(object):

    # Initialize a DataSet
    def __init__(self, config):
        self.config = config

    # shuffling rows and columns
    def permute(self, m):
        assert len(m.shape) == 2
        rowPermu = np.random.permutation(m.shape[0])
        colPermu = np.random.permutation(m.shape[1])
        return m[rowPermu, :][:, colPermu]

    # Read data
    def read_data(self, noise = True, train = True, lexSort = True):
        if noise:
            addr = self.config.input_dir_n + '/mn'
        else:
            addr = self.config.input_dir_p + '/mp'
        k = 0
        if train == True:
            inps = np.zeros((self.config.nTrain, self.config.nCells, self.config.nMuts), dtype = np.int8)
            for i in range(1, self.config.nTrain + 1):
                inp = np.genfromtxt(addr + '{}.txt'.format(i), skip_header=1, delimiter='\t')
                sol_matrix = np.delete(inp, 0, 1)
                if lexSort == True:
                    inps[k,:,:] = sol_matrix[:, np.lexsort(sol_matrix)]
                else:
                    inps[k,:,:] = self.permute(sol_matrix)
                k += 1
        else:
            inps = np.zeros((self.config.nTest, self.config.nCells, self.config.nMuts), dtype = np.int8)
            s_m = sample(range(self.config.nLow ,self.config.nHigh), self.config.nTest)
            for i in s_m:
                inp = np.genfromtxt(addr + '{}.txt'.format(i), skip_header=1, delimiter='\t')
                sol_matrix = np.delete(inp, 0, 1)
                if lexSort == True:
                    inps[k,:,:] = sol_matrix[:, np.lexsort(sol_matrix)]
                else:
                    inps[k,:,:] = self.permute(sol_matrix)
                k += 1       
        return inps

    def Cat_Shuf(self, train = True):
        m_n = self.read_data(noise = True, train = train, lexSort = True)
        m_p = self.read_data(noise = False, train = train, lexSort = True)
        if train == True:
            l_n = np.ones((self.config.nTrain, 1), dtype = np.int8)
            l_p = np.zeros((self.config.nTrain, 1), dtype = np.int8)
        else:
            l_n = np.ones((self.config.nTest, 1), dtype = np.int8)
            l_p = np.zeros((self.config.nTest, 1), dtype = np.int8)

        m = np.concatenate((m_n, m_p), axis = 0)
        # print(np.shape(m))
        if train == True:
            assert m.shape[0] == 2 * self.config.nTrain
        else:
            assert m.shape[0] == 2 * self.config.nTest
        l = np.concatenate((l_n, l_p), axis = 0)
        # print(np.shape(l))
        if train == True:
            assert l.shape[0] == 2 * self.config.nTrain
        else:
            assert l.shape[0] == 2 * self.config.nTest
        Permu = np.random.permutation(m.shape[0])
        m = m[Permu, :, :]
        l = l[Permu, :]
        return m, l

    def saveDataSet(self, X, y, fileName = None):
        if not fileName:
            fileName = f"dataset_{X.shape}_{y.shape}_{time()}.h5"
        saveingTime = time()
        fileAddress = os.path.join(self.config.h5_dir, fileName)
        print(fileAddress)
        h5f = h5py.File(fileAddress, 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('y', data=y)
        h5f.close()
        saveingTime = time() - saveingTime
        print(f"Input saved to {self.config.h5_dir} in time {saveingTime}")

    def loadDataSet(self, fileName):
        fileAddress = os.path.join(self.config.h5_dir, fileName)
        assert(os.path.exists(fileAddress))
        loadingTime = time()
        h5f = h5py.File(fileAddress, 'r')
        X = h5f['X'][:]
        y = h5f['y'][:]
        h5f.close()
        loadingTime = time() - loadingTime
        print(f"Input loaded in time {loadingTime}")
        return X, y

    # Generate random batch for testing procedure
    def test_batch(self, batch_size, nCells, nMuts, seed=0):
        # Generate random TSP instance
        pass
    
