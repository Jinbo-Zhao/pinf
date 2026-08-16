import numpy as np
from scipy.stats import t, f

def calc_cutoff(k, n_vec, alpha, discrep_string, random_state=None):
    """
    Fast vectorized version of calc_cutoff.
    Please refer to David Eckman's Plausible Screening (2020) for the details of the calculation.
    And flat chance for the gradient's cutoff calculation.
    Args:
        k: number of systems
        n_vec: vector of sample sizes for each system
        alpha: Type I error level (1-confidence level)
        discrep_string: type of discrepancy, one of ['ell1', 'ell2', 'ellinf', 'CRN', 'gradient']
        random_state: random state for reproducibility


    Returns:
        D_cutoff: cutoff value
    """
    rng = np.random.default_rng(random_state)
    # number of Monte Carlo repetitions
    # NOTE: n_MC_reps can be adjusted to trade off accuracy and computation time.
    n_MC_reps = 10**5

    # CRN has a closed-form cutoff and only needs a scalar sample size; handle it early.
    if discrep_string == 'CRN':
        n = float(np.asarray(n_vec).flat[0])
        D_cutoff = k * (n - 1) / (n - k) * f.ppf(1 - alpha, k, n - k)
        return D_cutoff

    n_vec = np.asarray(n_vec)
    n_vec = n_vec.reshape(-1, 1)  # for later broadcastings

    sample_size_length = np.size(n_vec)
    if sample_size_length != k:
        raise ValueError("The size of n_vec must be equal to k.")

    if discrep_string == 'gradient':
        #print("Gradient")
        dfn=2*np.ones((k,1))
        dfd=n_vec-2
        
        scale=2*(n_vec-1)/(n_vec-2)
        terms=scale*rng.f(dfn,dfd,size=(k,n_MC_reps))
        D_cutoff=np.quantile(np.sum(terms,axis=0),1-alpha)
        return D_cutoff

    elif discrep_string == 'ell1':
        # Vectorized t draws: shape = (k, n_MC_reps)
        df = n_vec - 1
        terms = rng.standard_t(df, size=(k, n_MC_reps))
        stats = np.sum(np.abs(terms), axis=0)
        D_cutoff = np.quantile(stats, 1 - alpha)

    elif discrep_string == 'ell2':
        # Vectorized F draws
        dfn = np.ones((k, 1))
        dfd = n_vec - 1
        terms = rng.f(dfn, dfd, size=(k, n_MC_reps))
        #terms=f.rvs(dfn, dfd, size=(k, n_MC_reps), random_state=rng)
        stats = np.sum(terms, axis=0)
        D_cutoff = np.quantile(stats, 1 - alpha)

    elif discrep_string == 'ellinf':
        # Vectorized max-abs t draws
        df = n_vec - 1
        terms = rng.standard_t(df, size=(k, n_MC_reps))
        #terms=t.rvs(df, size=(k, n_MC_reps), random_state=rng)
        stats = np.max(np.abs(terms), axis=0)
        D_cutoff = np.quantile(stats, 1 - alpha)

    else:
        raise ValueError("Specify a valid discrepancy: {'ell1', 'ell2', 'ellinf', 'CRN', 'gradient'}.")

    return D_cutoff
