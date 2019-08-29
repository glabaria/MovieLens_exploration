'''
Author: George Labaria
References: heavily influenced by: 
    https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea

This class computes the explicit matrix factorization approximation to a rating
matrix for the purpose of recommendation systems.  Let R be the rating matrix,
with n_u users and n_i items (n_u x n_i).  Let k be the number of features for
each user and each item.  Then R ~ XY where X is n_u x k and Y is k x n_i.

This class computes the matrix factorization using one of two methods:
    (1) alternating least squares (als)
    (2) stochastic gradient descent (sgd)
    
initial parameters:
    ratings: ratings matrix (users x items)
    n_factors: number of factors (features) for each user and each item
    item_reg: regularization constant for the items
    user_reg: regularization constant for the users
    item_bias_reg: bias regularization constant for the items (sgd only)
    user_bias_reg: bias regularization constant for the users (sgd only)
    alg: flag to use als or sgd must be 'als' or 'sgd' (string)
'''

import numpy as np
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

class ExplicitMF(object):
    def __init__(self,
                 ratings,
                 n_factors=40,
                 item_reg=0.0,
                 user_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 alg='sgd',
                 implicit_pref = False):
        
        self.ratings = ratings
        self.n_users,self.n_items = ratings.shape
        self.item_reg = item_reg
        self.user_reg = user_reg
        self.n_factors = n_factors
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.alg = alg
        self.implicit_pref = implicit_pref
        
        if self.alg == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
                
    def ALS_step(self,
                 latent,
                 fixed,
                 ratings,
                 lamb,
                 _type='user'):
        
        if _type == 'user':
            YTY = fixed.T.dot(fixed)
            lambdaI = np.eye(YTY.shape[0])*lamb
            
            for i in range(latent.shape[0]):
                latent[i,:] = solve(YTY+lambdaI,ratings[i,:].dot(fixed))
                
        elif _type == 'item':
            XTX = fixed.T.dot(fixed)
            lambdaI = np.eye(XTX.shape[0])*lamb
            
            for i in range(latent.shape[0]):
                latent[i,:] = solve(XTX+lambdaI,ratings[:,i].T.dot(fixed))
                
        return latent
                
    def train(self, n_iter, init_flag=True, learning_rate=0.01):
        if init_flag == True:    
            self.user_mat = np.random.random((self.n_users,self.n_factors))
            self.item_mat = np.random.random((self.n_items,self.n_factors))
        
            if self.alg == 'sgd':
                self.learning_rate = learning_rate
                self.user_bias = np.zeros(self.n_users)
                self.item_bias = np.zeros(self.n_items)
                self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
                
                if self.implicit_pref == True:
                    #this is the extra item factors for the implicit preference
                    self.extra_item_factors = np.random.random((self.n_items,self.n_factors))
                    self.build_ipm()
            
        
        if self.alg == 'als':
            for i in range(n_iter):
                self.user_mat = self.ALS_step(self.user_mat,
                                              self.item_mat,
                                              self.ratings,
                                              self.user_reg,
                                              _type='user')
            
                self.item_mat = self.ALS_step(self.item_mat,
                                              self.user_mat,
                                              self.ratings,
                                              self.item_reg,
                                              _type='item')
        elif self.alg == 'sgd' and self.implicit_pref == False:

            for i in range(n_iter):
                self.sgd()
                
        elif self.alg == 'sgd' and self.implicit_pref == True:
            
            for i in range(n_iter):
                #iterate through sgd w/ implicit pref
                self.sgd_ip()        
        
    def sgd(self):
        self.training_ind = np.arange(self.n_samples)
        np.random.shuffle(self.training_ind)
        
        for ind in self.training_ind:
            u = self.sample_row[ind]
            i = self.sample_col[ind]
            
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i] + \
                        self.user_mat[u, :].dot(self.item_mat[i, :].T)
            err = self.ratings[u,i]-prediction
            
            #print(err)
            
            #update biases
            self.user_bias[u] += self.learning_rate * \
                                (err - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * \
                                (err - self.item_bias_reg * self.item_bias[i])
                                
            #update latent factors
            self.user_mat[u, :] += self.learning_rate * \
                                    (err * self.item_mat[i, :] - \
                                     self.user_reg * self.user_mat[u,:])
            self.item_mat[i, :] += self.learning_rate * \
                                    (err * self.user_mat[u, :] - \
                                     self.item_reg * self.item_mat[i,:])
    
    #stochastic grad descent w/ implicit preferences               
    def sgd_ip(self):
        self.training_ind = np.arange(self.n_samples)
        np.random.shuffle(self.training_ind)
        numN = len(self.ipm)
        
        for ind in self.training_ind:
            u = self.sample_row[ind]
            i = self.sample_col[ind]
            
            curr_sum_ipm = self.sum_ipm(u)
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i] + \
                        self.item_mat[i, :].T.dot(self.user_mat[u, :] + \
                        1.0/np.sqrt(numN) * curr_sum_ipm)
            err = self.ratings[u,i]-prediction
            
            #update biases
            self.user_bias[u] += self.learning_rate * \
                                (err - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * \
                                (err - self.item_bias_reg * self.item_bias[i])
                                
            #update latent factors
            self.user_mat[u, :] += self.learning_rate * \
                                    (err * self.item_mat[i, :] - \
                                     self.user_reg * self.user_mat[u,:])
            self.item_mat[i, :] += self.learning_rate * \
                                    (err * (self.user_mat[u, :] + \
                                    1.0/np.sqrt(numN)*curr_sum_ipm) - \
                                     self.item_reg * self.item_mat[i,:])
                                    
            self.extra_item_factors[i,:] += self.learning_rate * \
                                            1.0/np.sqrt(numN)*self.sum_err_along_items(u) - \
                                            self.item_reg * self.extra_item_factors[i,:]
                                    
    def predict(self):
        prediction = np.zeros((self.user_mat.shape[0],self.item_mat.shape[0]))
        
        for u in range(self.user_mat.shape[0]):
            for i in range(self.item_mat.shape[0]):
                
                if self.alg == 'als':
                    prediction[u,i] = self.user_mat[u,:].dot(self.item_mat[i,:].T)
                elif self.alg == 'sgd':
                    if self.implicit_pref == False:
                        prediction[u,i] = self.global_bias + self.user_bias[u] + self.item_bias[i] + \
                            self.user_mat[u, :].dot(self.item_mat[i, :].T)
                    else:
                        curr_sum_ipm = self.sum_ipm(u)
                        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i] + \
                                    self.item_mat[i, :].T.dot(self.user_mat[u, :] + \
                                    1.0/np.sqrt(len(self.ipm)) * curr_sum_ipm)
                        
                    #print('(rating,pred)=',self.ratings[u,i],prediction[u,i])
                
        return prediction
    
    def learning_curve(self, iter_list, test, learning_rate=0.01):
        
        assert(len(iter_list) > 1)
        
        iter_list.sort()
        test_mse = []
        train_mse = []
        
        prev_iter = 0
        for i in range(len(iter_list)):
            curr_iter = iter_list[i]
            if i == 0:
                self.train(curr_iter, init_flag=True, learning_rate=learning_rate)
            else:
                self.train(curr_iter-prev_iter,init_flag=False, learning_rate=learning_rate)
                
            prediction = self.predict()
            test_mse.append(self.get_mse(prediction,test))
            train_mse.append(self.get_mse(prediction,self.ratings))
            
            prev_iter = curr_iter
            
        return test_mse,train_mse
        
    def get_mse(self, prediction, observed):
        
        prediction = prediction[observed.nonzero()].flatten()
        observed = observed[observed.nonzero()].flatten()
        
        return mean_squared_error(observed,prediction)
    
    def build_ipm(self):
        
        #implicit preference matrix
        self.ipm = {}
        
        #if user gave rating 4 or 5, the user displays preference for that movie
        for u in range(self.n_users):
            for i in range(self.n_items):
                if self.ratings[u,i] >= 4:
                    self.ipm[(u,i)] = np.random.random(self.n_factors)
                    
    #computes sum_{j \in N(u)} x_j where N(u) is the set of items that user u implictly prefers
    def sum_ipm(self,u):
        
        result = np.zeros(self.n_factors)
        for key in self.ipm:
            if key[0] == u:
                result += self.ipm[key]
                
        return result
                
    #computes the sum_{i} e_{ui}*q_i where e_{ui} is the error for user u and item i
    def sum_err_along_items(self, u):
        
        result = np.zeros(self.n_factors)
        for ind in self.training_ind:
            i = self.sample_col[ind]
            
            
            curr_sum_ipm = self.sum_ipm(u)
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i] + \
                        self.item_mat[i, :].T.dot(self.user_mat[u, :] + \
                        1.0/np.sqrt(len(self.ipm)) * curr_sum_ipm)
            err = self.ratings[u,i]-prediction
            result += err*self.item_mat[i,:]
            
        return result