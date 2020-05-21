import numpy as np
import pandas as pd
import math
from sympy import *
import matplotlib.pyplot as plt

def generate_std_data(n,m,std):
    num_samples = n
    desired_mean = m
    desired_std_dev = std

    samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=num_samples)

    actual_mean = np.mean(samples)
    actual_std = np.std(samples)
    print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))

    zero_mean_samples = samples - (actual_mean)

    zero_mean_mean = np.mean(zero_mean_samples)
    zero_mean_std = np.std(zero_mean_samples)
    print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))

    scaled_samples = zero_mean_samples * (desired_std_dev / zero_mean_std)
    scaled_mean = np.mean(scaled_samples)
    scaled_std = np.std(scaled_samples)
    print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))

    final_samples = scaled_samples + desired_mean
    final_mean = np.mean(final_samples)
    final_std = np.std(final_samples)
    return final_samples

def generate_random_correlated_walks(decomposition, x, n):
    uncorrelated_walks = np.random.normal(loc=0, scale=1, size=(2*x, n))
    correlated_walks = np.dot(decomposition.T, uncorrelated_walks)
    return correlated_walks

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
def generate_random_correlated_walks(decomposition, x, n):
    uncorrelated_walks = np.random.normal(loc=0, scale=1, size=(2*x, n))
    correlated_walks = np.dot(decomposition.T, uncorrelated_walks)
    return correlated_walks
def symbolize(s):
    """
    Converts a a string (equation) to a SymPy symbol object
    """
    from sympy import sympify
    s1 = s.replace('.', '*')
    s2 = s1.replace('^', '**')
    s3 = sympify(s2)

    return (s3)
def eval_multinomial(s, vals=None, symbolic_eval=False):
    """
    Evaluates polynomial at vals.
    vals can be simple list, dictionary, or tuple of values.
    vals can also contain symbols instead of real values provided those symbols have been declared before using SymPy
    """
    from sympy import Symbol
    sym_s = symbolize(s)
    sym_set = sym_s.atoms(Symbol)
    sym_lst = []
    for s in sym_set:
        sym_lst.append(str(s))
    sym_lst.sort()
    if symbolic_eval == False and len(sym_set) != len(vals):
        print("Length of the input values did not match number of variables and symbolic evaluation is not selected")
        return None
    else:
        if type(vals) == list:
            sub = list(zip(sym_lst, vals))
        elif type(vals) == dict:
            l = list(vals.keys())
            l.sort()
            lst = []
            for i in l:
                lst.append(vals[i])
            sub = list(zip(sym_lst, lst))
        elif type(vals) == tuple:
            sub = list(zip(sym_lst, list(vals)))
        result = sym_s.subs(sub)

    return result
def gen_regression_symbolic(m=None, n_samples=100, n_features=2, noise=0.0, noise_dist='normal'):
    """
    Generates regression sample based on a symbolic expression. Calculates the output of the symbolic expression
    at randomly generated (drawn from a Gaussian distribution) points
    m: The symbolic expression. Needs x1, x2, etc as variables and regular python arithmatic symbols to be used.
    n_samples: Number of samples to be generated
    n_features: Number of variables. This is automatically inferred from the symbolic expression. So this is ignored
                in case a symbolic expression is supplied. However if no symbolic expression is supplied then a
                default simple polynomial can be invoked to generate regression samples with n_features.
    noise: Magnitude of Gaussian noise to be introduced (added to the output).
    noise_dist: Type of the probability distribution of the noise signal.
    Currently supports: Normal, Uniform, t, Beta, Gamma, Poission, Laplace

    Returns a numpy ndarray with dimension (n_samples,n_features+1). Last column is the response vector.
    """

    import numpy as np
    from sympy import Symbol, sympify

    if m == None:
        m = ''
        for i in range(1, n_features + 1):
            c = 'x' + str(i)
            c += np.random.choice(['+', '-'], p=[0.5, 0.5])
            m += c
        m = m[:-1]

    sym_m = sympify(m)
    n_features = len(sym_m.atoms(Symbol))
    evals = []
    lst_features = []

    for i in range(n_features):
        lst_features.append(np.random.normal(scale=5, size=n_samples))
    lst_features = np.array(lst_features)
    lst_features = lst_features.T
    lst_features = lst_features.reshape(n_samples, n_features)

    for i in range(n_samples):
        evals.append(eval_multinomial(m, vals=list(lst_features[i])))

    evals = np.array(evals)
    evals = evals.reshape(n_samples, 1)

    if noise_dist == 'normal':
        noise_sample = noise * np.random.normal(loc=0, scale=1.0, size=n_samples)
    elif noise_dist == 'uniform':
        noise_sample = noise * np.random.uniform(low=0, high=1.0, size=n_samples)
    elif noise_dist == 'beta':
        noise_sample = noise * np.random.beta(a=0.5, b=1.0, size=n_samples)
    elif noise_dist == 'Gamma':
        noise_sample = noise * np.random.gamma(shape=1.0, scale=1.0, size=n_samples)
    elif noise_dist == 'laplace':
        noise_sample = noise * np.random.laplace(loc=0.0, scale=1.0, size=n_samples)

    noise_sample = noise_sample.reshape(n_samples, 1)
    evals = evals + noise_sample

    x = np.hstack((lst_features, evals))

    return (x)
num_samples = 400

# The desired mean values of the sample.
mu = np.array([10,100 , 50.0,30])

# The desired covariance matrix.
r = np.array([[ 1.        ,  0.78886583,  0.70198586,  0.56810058],
       [ 0.78886583,  1.        ,  0.49187904,  0.45994833],
       [ 0.70198586,  0.49187904,  1.        ,  0.4755558 ],
       [ 0.56810058,  0.45994833,  0.4755558 ,  1.        ]])
# Generate the random samples.
y = np.random.multivariate_normal(mu, r, size=20000)
df=pd.DataFrame(y,columns=['q','w','e','r'])
# df_1=pd.DataFrame(y,columns=['q','w','e','r'])
# df.sort_values(by='q', inplace=True)
# f_col = [1.3*x**2+1.2*x for x in df['q']]
# df = df.assign(f = f_col ).sort_values(by='f').drop('f', axis=1)
# f_col = [x**2*exp(-0.5*x)*sin(x+10)  for x in df['w']]
# df = df.assign(f = f_col ).sort_values(by='f').drop('f', axis=1)
f_col = [x**2*exp(-0.5*x)*sin(x+10) for x in df['e']]
df = df.assign(f = f_col ).sort_values(by='f').drop('f', axis=1)
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)
# x = gen_regression_symbomalic(m='x**2*exp(-0.5*x)*sin(x+10)',n_samples=50,noise=1)