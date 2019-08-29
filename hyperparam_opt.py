import numpy as np
import MF

def hyperparam_grid_search(i,iter_list,test,train,fact_vect,reg_vect):
    m = len(reg_vect)
    
    test_mse_mat = np.zeros(m)
    
    for j in range(m):
        print('i = ',i,'; j = ',j)
        
        SGD_MF = MF.ExplicitMF(train,n_factors=fact_vect[i],item_reg=reg_vect[j],user_reg=reg_vect[j],item_bias_reg=reg_vect[j],user_bias_reg=reg_vect[j],alg='sgd')
        test_mse,train_mse = SGD_MF.learning_curve(iter_list,test,learning_rate=0.01)
        
        opt_iter_ind = np.argmin(test_mse)
        
        test_mse_mat[j] = test_mse[opt_iter_ind]
        
    ind_opt =  np.unravel_index(np.argmin(test_mse_mat),test_mse_mat.shape)
    
    return (i, ind_opt, test_mse_mat[ind_opt], iter_list[opt_iter_ind])