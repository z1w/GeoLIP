import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
import multiprocessing as mp
import scipy.sparse as sp

class GL_Solver:
    def __init__(self, weights, norm='inf', dual=True, approx_hidden=False, approx_input=False):
        self.weight_mats = weights
        self.norm=norm
        if not self.norm == 'inf':
            raise NotImplementedError
        self.layers=len(weights)
        self.weight_dims = [0] * self.layers
        for i in range(self.layers):
            self.weight_dims[i] = self.weight_mats[i].shape[1]
        #nClasses is the number of classification classes, e.g., nClasses=10 for CIFAR10
        self.nClasses = self.weight_mats[-1].shape[0]
        if self.layers > 2 or dual:
            self.dual=True
        else:
            self.dual=False
        if self.dual:
            self.construct_mat()
        self.approx_hidden = approx_hidden
        self.approx_input = approx_input
            

    def construct_mat(self):
        #n_hidden_vars is the number of hidden nodes
        self.n_hidden_vars = sum(self.weight_dims[1:])
        #Constructing some auxilary matrices for the SDP program
        weights = block_diag(*self.weight_mats[:-1])
        zeros_col = np.zeros((weights.shape[0], self.weight_dims[-1]))
        A = np.concatenate((weights, zeros_col), axis=1)
        eyes = np.identity(A.shape[0])
        init_col = np.zeros((eyes.shape[0], self.weight_dims[0]))
        B = np.concatenate((init_col, eyes), axis=1)
        A_on_B = np.concatenate((A, B), axis = 0)
        extra_col = np.zeros((A_on_B.shape[0], 1))
        self.mult = np.concatenate((extra_col, A_on_B), axis=1)
        self.constraint_size = self.mult.shape[1]


    def solve_dual_program(self, final_weight):
        L_sq = cp.Variable((1,1), nonneg=True)
        if self.approx_hidden:
            Layer_D = []
            for i in range(self.layers-1):
                Layer_D.append(cp.Variable(nonneg=True))
            #cp.Variable((1, (self.layers-1)), nonneg=True)
            D_list = []
            nVars = 0
            for i in range(self.layers-1):
                nVars += self.weight_dims[i+1]
                D_list.append(np.ones((1, self.weight_dims[i+1]))*Layer_D[i])
            assert nVars==self.n_hidden_vars
            D = cp.hstack(D_list)
        else:
            D = cp.Variable((1, self.n_hidden_vars), nonneg=True)
        if self.approx_input:
            DualIN=np.ones((self.weight_dims[0], 1))*cp.Variable(nonneg=True)
        else:
            DualIN = cp.Variable((self.weight_dims[0], 1), nonneg=True)
        T = cp.diag(D)
        Q = cp.bmat([[0*T, T],[T, -2*T]])
        const_matrix = self.mult.transpose() @ Q @ self.mult
        #Create Sparse Diagonal Variable Matrix
        obj_term = L_sq-cp.sum(DualIN)
        sparse_vars = cp.vstack([obj_term, DualIN])
        positions = []
        for i in range(self.weight_dims[0]+1):
            positions.append([i, i])
        #assert len(range(1, self.weight_dims[0]+1)) == self.weight_dims[0]
        V = np.ones(self.weight_dims[0]+1)
        I = []
        J = []
        for idx, (row, col) in enumerate(positions):
            I.append(row + col*self.constraint_size)
            J.append(idx)
        reshape_mat = sp.coo_matrix((V, (I,J)), shape=(self.constraint_size*self.constraint_size, self.weight_dims[0]+1))
        M = cp.reshape(reshape_mat @ sparse_vars, (self.constraint_size,self.constraint_size))
        #Another Matrix
        N = np.zeros([self.constraint_size,self.constraint_size])
        N[0, self.constraint_size-self.weight_dims[-1]:] = -final_weight
        N[self.constraint_size-self.weight_dims[-1]:, 0] = -final_weight
        #The CVX optimization program
        prob = cp.Problem(cp.Minimize(L_sq), [(self.mult.transpose() @ Q @ self.mult) - M - N << 0])
        #Verbose: False if not want to print out the progress from the solver
        #prob.solve(solver=getattr(cp, 'SCS'), verbose=True, **{'gpu': True, 'use_indirect': True, 'eps_abs':1.0, 'max_iters':500})
        prob.solve(solver=cp.MOSEK, verbose=True)

        return prob.value
    
    def solve_primal_program(self, final_weight):
        prop_weight = np.matmul(self.weight_mats[0].T, np.diagflat(final_weight))
        one_shape = prop_weight.shape[1]
        ones = np.ones((one_shape, 1))
        #print(np.matmul(prop_weight, ones).shape)
        #print(prop_weight.shape)
        sdp_weight_entries = np.concatenate((prop_weight, np.matmul(prop_weight, ones)), axis = 1)
        entries_shape = sdp_weight_entries.shape
        sdp_weight_shape = entries_shape[0]+entries_shape[1]
        sdp_weight = np.zeros((sdp_weight_shape, sdp_weight_shape))
        sdp_weight[:entries_shape[0], entries_shape[0]:] = sdp_weight_entries

        X = cp.Variable((sdp_weight_shape, sdp_weight_shape), PSD=True)
        #The CVX optimization program
        prob = cp.Problem(cp.Maximize(cp.trace(sdp_weight @ X)), [cp.diag(X) == np.ones(sdp_weight_shape)])
        #Verbose: False if not want to print out the progress from the solver
        #prob.solve(solver=getattr(cp, 'SCS'), verbose=True, **{'gpu': True, 'use_indirect': True})
        prob.solve(solver=cp.MOSEK, verbose=True)

        return prob.value

    def single_processing_norm(self, i):
        final_weight = self.weight_mats[-1][i,:]
        if self.dual:
            val = self.solve_dual_program(final_weight)/2
        else:
            val = self.solve_primal_program(final_weight)/2
        return ["class: "+str(i), val]

    def sdp_norm(self, parallel=True):
        if parallel:
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap(self.single_processing_norm, [(i,) for i in range(self.nClasses)])
            pool.close()
        else:
            results = []
            for i in range(self.nClasses):
                results.append(self.single_processing_norm(i))
        return results
