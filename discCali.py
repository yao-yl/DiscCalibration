import tensorflow
import tensorflow_probability
import jax
print(f"{jax.__version__=}")
print(f"{tensorflow.__version__=}")
print(f"{tensorflow_probability.__version__=}")
import tensorflow_probability.substrates.jax as tfp
from jax import config
config.update("jax_enable_x64", True)
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache(f"jax-compilation-cache")
from jax import numpy as np
from matplotlib import pyplot as plt
import numpy as onp
import time
import jax.flatten_util
import jax.scipy.optimize
from sklearn.model_selection import KFold

M = 10 # number of inference draws

def gendata(key):
    S = 100 # number of simulations
    dims = 2

    # probabilistic model
    # θ ~ N(0,1)
    # y ~ N(θ,s)

    s = 0.1 # prior scale

    delta = 2 # approximation error

    # true parameters
    key,subkey = jax.random.split(key)
    theta = jax.random.normal(subkey,shape=[dims,S])
    # data
    key,subkey = jax.random.split(key)
    y = theta + s*jax.random.normal(subkey,shape=[dims,S])
    # true posterior draws
    true_mean = y/s**2/(1 + 1/s**2)
    true_scale = np.sqrt(1/(1 + 1/s**2))
    # approximate posterior draws
    key,subkey = jax.random.split(key)
    inf_theta = true_mean[:,:,None] + delta*true_scale*jax.random.normal(subkey,shape=[dims,S,M])

    return theta, inf_theta, y

def getrank(theta,inf_theta):
    rank = np.sum(theta[:,:,None]>inf_theta,2)
    return rank

def get_binary_data(theta,inf_theta,y):
    dims,S = theta.shape
    dims2,S2,M = inf_theta.shape
    y_dims,S3 = y.shape
    assert dims==dims2
    assert S==S2
    assert S==S3
    t = onp.zeros(S*(M+1))
    phi = onp.zeros([S*(M+1),dims+y_dims])
    weight = onp.zeros(S*(M+1))
    where = 0
    for s in range(S):
        t[where] = 1
        phi[where] = np.concatenate([y[:,s],theta[:,s]])
        weight[where] = M
        where += 1
        t[where:where+M] = 0
        big_y = np.repeat(y[:,s,np.newaxis], inf_theta.shape[2], axis=-1)
        phi[where:where+M] = np.concatenate([big_y,inf_theta[:,s,:]],axis=0).T
        weight[where:where+M] = 1
        where += M
    weight = np.array(weight) / np.sum(weight)
    return np.array(t),np.array(phi), weight

def binary_diagnostic(t,phi,weight,t_val,phi_val,weight_val,key,regularization):
    weight = weight / np.sum(weight)
    weight_val = weight_val / np.sum(weight_val)

    def pointwise_loss(params,t,phi):
        b,w,c,U = unflatten(params)
        pred = b + np.tanh(c + phi @ U) @ w
        loss = jax.nn.softplus(-pred * (2*t-1))
        return loss

    def loss(params,t,phi,weight):
        return pointwise_loss(params,t,phi) @ weight + regularization*np.sum(params**2)

    loss_train = jax.jit(lambda params: loss(params,t,phi,weight))

    n_hidden = 32
    b = 0.0
    key,subkey = jax.random.split(key)
    w = .01*jax.random.normal(subkey,[n_hidden])
    key,subkey = jax.random.split(key)
    c = .01*jax.random.normal(subkey,[n_hidden])
    key,subkey = jax.random.split(key)
    U = 1*jax.random.normal(subkey,[phi.shape[1],n_hidden])

    params, unflatten = jax.flatten_util.ravel_pytree([b,w,c,U])

    prob0 = 1/2
    neg_entropy = prob0*np.log(prob0) + (1-prob0)*np.log(1-prob0)

    obj = jax.jit(jax.value_and_grad(loss_train))
    rez = tfp.optimizer.lbfgs_minimize(obj,initial_position=params,max_iterations=1000)
    num_iters = rez.num_iterations
    final_loss = rez.objective_value
    params = rez.position
    solved = rez.converged
    print(f"{num_iters=} {solved=}")
    #assert solved
    assert num_iters > 5

    Ds_train = -pointwise_loss(params,t,phi) - neg_entropy
    Ds_val = -pointwise_loss(params,t_val,phi_val) - neg_entropy

    def weighted_std(X,W):
        assert X.shape==W.shape
        assert np.abs(np.sum(W)-1.0) < 1e-5
        var = np.sum(X**2 * W) - np.sum(X * W)**2
        std = np.sqrt(var)
        return std

    D_train = Ds_train @ weight
    D_train_std = weighted_std(Ds_train,weight)
    D_val = Ds_val @ weight_val
    D_val_std = weighted_std(Ds_val,weight_val)
    return D_train, D_train_std, D_val, D_val_std

def cv(diagnostic,train_data, test_data, key):
    # split training data into folds
    kf = KFold(n_splits=2)
    # for each fold train with variety of regularizers, store val loss
    regs = [.01, .001,.0001,.00001]
    val_D = onp.zeros([len(regs),kf.get_n_splits(train_data[0])])
    for i, reg in enumerate(regs):
        for j, (train_index, val_index) in enumerate(kf.split(train_data[0])):
            train = [a[train_index] for a in train_data]
            val = [a[val_index] for a in train_data]

            val_D[i,j] = diagnostic(*train, *val,key,reg)[2]
            print(val_D)

    # pick best regularizer
    mean_val_D = np.mean(val_D,axis=1)
    best_reg = regs[np.argmax(mean_val_D)]
    print(val_D)
    print(f"{best_reg=}")
    
    # retain on all data
    return diagnostic(*train_data,*test_data,key,best_reg)

def test_binary():
    key = jax.random.PRNGKey(0)
    key,subkey = jax.random.split(key)
    data = gendata(key)
    key,subkey = jax.random.split(key)
    val_data = gendata(key)

    t,phi,weight = get_binary_data(*data)
    t_val,phi_val,weight_val = get_binary_data(*val_data)
    D_mean, D_std, D_mean_val, D_std_val = cv(binary_diagnostic,[t,phi,weight],[t_val,phi_val,weight_val],key)
    print(f"{D_mean=} {D_std=} {D_mean_val=} {D_std_val=}")


def get_multiclass_data(theta,inf_theta,y):
    dims,S = theta.shape
    dims2,S2,M = inf_theta.shape
    y_dims,S3 = y.shape
    assert dims==dims2
    assert S==S2
    assert S==S3
    t = onp.zeros(S)
    phi = onp.zeros([S,M+1,dims+y_dims])
    where = 0
    for s in range(S):
        t[where] = 0
        phi[where,0] = np.concatenate([y[:,s],theta[:,s]])
        big_y = np.repeat(y[:,s,np.newaxis], inf_theta.shape[2], axis=-1)
        phi[where,1:,:] = np.concatenate([big_y,inf_theta[:,s,:]],axis=0).T
        where += 1
    return np.array(t),np.array(phi)


# We code the  separable classifier (Eq 12) in the paper. Gamma is all linear features, such as log q(theta|y).

def multiclass_diagnostic(t,phi,gamma,t_val,phi_val,gamma_val,key,regularization):

    # run network on 1 channel
    def net1(b,w,c,U,v,phi1,gamma1):
        return b + np.tanh(c + phi1 @ U) @ w + gamma1 @ v

    net = jax.vmap(net1,[None,None,None,None,None,1,1])

    def pointwise_loss(params,t,phi,gamma):
        b,w,c,U,v = unflatten(params)
        pred = net(b,w,c,U,v,phi,gamma)
        
        return -(pred[0] - jax.nn.logsumexp(pred,axis=0))

    def loss(params,t,phi,gamma):
        return np.mean(pointwise_loss(params,t,phi,gamma)) + regularization*np.sum(params**2)

    loss_train = jax.jit(lambda params: loss(params,t,phi,gamma))

    n_hidden = 32
    b = 0.0
    key,subkey = jax.random.split(key)
    w = .01*jax.random.normal(subkey,[n_hidden])
    key,subkey = jax.random.split(key)
    c = .01*jax.random.normal(subkey,[n_hidden])
    key,subkey = jax.random.split(key)
    U = 1*jax.random.normal(subkey,[phi.shape[2],n_hidden])
    key,subkey = jax.random.split(key)
    v = .01*jax.random.normal(subkey,[gamma.shape[2]])

    params, unflatten = jax.flatten_util.ravel_pytree([b,w,c,U,v])

    prob0 = 1/(M+1)
    neg_entropy = np.log(prob0)

    obj = jax.jit(jax.value_and_grad(loss_train))
    rez = tfp.optimizer.lbfgs_minimize(obj,initial_position=params,max_iterations=1000)
    num_iters = rez.num_iterations
    final_loss = rez.objective_value
    params = rez.position
    solved = rez.converged
    print(f"{num_iters=} {solved=}")
    #assert solved
    assert num_iters > 3

    Ds_train = -pointwise_loss(params,t,phi,gamma) - neg_entropy
    Ds_val = -pointwise_loss(params,t_val,phi_val,gamma_val) - neg_entropy

    D_train = np.mean(Ds_train)
    D_train_std = np.std(Ds_train)
    D_val = np.mean(Ds_val)
    D_val_std = np.std(Ds_val)
    return D_train, D_train_std, D_val, D_val_std

def test_multiclass():
    key = jax.random.PRNGKey(0)
    key,subkey = jax.random.split(key)
    data = gendata(key)
    key,subkey = jax.random.split(key)
    val_data = gendata(key)

    t,phi = get_multiclass_data(*data)
    #gamma = phi
    gamma = np.zeros([phi.shape[0],M+1,1])
    t_val,phi_val = get_multiclass_data(*val_data)
    #gamma_val = phi_val
    gamma_val = np.zeros([phi_val.shape[0],M+1,1])
    
    #D_mean, D_std, D_mean_val, D_std_val = multiclass_diagnostic(t,phi,t_val,phi_val,key,.001)
    D_mean, D_std, D_mean_val, D_std_val = cv(multiclass_diagnostic,[t,phi,gamma],[t_val,phi_val,gamma_val],key)
    print(f"{D_mean=} {D_std=} {D_mean_val=} {D_std_val=}")


## Experiment 1: Gaussian example, the following code generates binary multi-class label and then runs the corresponding classifiers:  
# test_multiclass()

# test_binary()

## Experiment 2: real data example:

sbidata = np.load('data/galaxy.npz')  # please unzip first 
# Here we use a real data example from cosmology. 
# Because of the file size limit of Github, we only upload a sub-sampled version of the actual data file in the repo.
print({sbidata['samples'].shape}) 
print({sbidata['x_val'].shape})
print({sbidata['y_val'].shape})
print({sbidata['logprob'].shape})
print({sbidata['samples_logprob'].shape})


# For illustration, we use S=1000, M=500 in the subsampled dataset. The index has been randomized. 
# Tips: we run the following code on a 64gb memory laptop. Given the relatively small example size (~1M) in the dataset, we are able to run LBFGS. 
#          If your machine has a much smaller memory, you may have memory issue and need to change to batch optimization, or use smaller M and S. 

S=1000
M=500
def reshapesbi(sbidata):
    INFTHETA=np.transpose(np.concatenate([sbidata['samples'], sbidata['samples_logprob'][:,:,None]], axis=2), (2,0,1))
    THETA=np.transpose(np.concatenate([sbidata['y_val'], sbidata['logprob'][:,None]], axis=1) , (1,0))
    Y=np.transpose(sbidata['x_val'], (1,0))
    inf_theta=INFTHETA [ :,0:S, 0:M]  
    y= Y[ :,0:S]
    theta= THETA[ :,0:S]
    return theta, inf_theta, y

def reshapesbi_val(sbidata):
    INFTHETA=np.transpose(np.concatenate([sbidata['samples'], sbidata['samples_logprob'][:,:,None]], axis=2), (2,0,1))
    THETA=np.transpose(np.concatenate([sbidata['y_val'], sbidata['logprob'][:,None]], axis=1) , (1,0))
    Y=np.transpose(sbidata['x_val'], (1,0))
    inf_theta=INFTHETA[ :,S:(S+S), 0:M]
    y= Y[ :,S:(S+S)]
    theta= THETA[ :,S:(S+S)]
    return theta, inf_theta, y

key = jax.random.PRNGKey(0)
data = reshapesbi(sbidata)
val_data = reshapesbi_val(sbidata)
t,phi,weight = get_binary_data(*data)
t_val,phi_val,weight_val = get_binary_data(*val_data)
# we have run cv, and choose the best regulariztion = 0.001. 
#To run CV, you can use:  D_mean, D_std, D_mean_val, D_std_val = cv(binary_diagnostic,[t,phi,weight],[t_val,phi_val,weight_val],key)
D_mean, D_std, D_mean_val, D_std_val = binary_diagnostic(t,phi,weight,t_val,phi_val,weight_val,key,regularization=.001)

# estimated divergence:
print(D_mean_val)


