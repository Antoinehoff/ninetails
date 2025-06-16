from scipy.special import factorial
import numpy as np

def ntopj(n):
    if n == 0: return (0, 0)
    if n == 1: return (1, 0)
    if n == 2: return (2, 0)
    if n == 3: return (0, 1)
    if n == 4: return (3, 0)
    if n == 5: return (1, 1)
    if n == 6: return (4, 0)
    if n == 7: return (2, 1)
    if n == 8: return (0, 2)

def non_adiab_moment_mapping(model,y,p,j):
    # return the p,j non adiabatic moment
    nadiab = 1.0/model.p.tau * kernel(model,j)*(y[-1]) if p == 0 \
        else 0.0
    n = -1
    if (p,j) == (0,0): n = 0
    if (p,j) == (1,0): n = 1
    if (p,j) == (2,0): n = 2
    if (p,j) == (0,1): n = 3
    if (p,j) == (3,0): n = 4
    if (p,j) == (1,1): n = 5
    if (p,j) == (4,0): n = 6
    if (p,j) == (2,1): n = 7
    if (p,j) == (0,2): n = 8
    if n > 0:
        return y[n] + nadiab
    else:
        return model.zeros

def kernel(model, j):
    if j == 0: return model.K0
    if j == 1: return model.K1
    if j == 2: return model.K2
    return model.zeros

def Mparapj(model, y, p, j):
    Mna = lambda p,j: non_adiab_moment_mapping(model, y, p, j)

    curlyNpm1j = np.sqrt(p+1) * Mna(p+1,j) + np.sqrt(p) * Mna(p-1,j)
    curlyNpm1jm1 = np.sqrt(p+1) * Mna(p+1,j-1) + np.sqrt(p) * Mna(p-1,j-1)
    # Here Cpar = sqrt(tau)/sigma/Jacobian/hatB ddz
    return np.sqrt(model.p.tau) * ( model.Cpar(curlyNpm1j) - model.CparB((j+1) * curlyNpm1j - j * curlyNpm1jm1)) \
                                  + np.sqrt(p) * model.CparB( (2*j+1) * Mna(p-1,j) - (j+1) * Mna(p-1,j+1) - j * Mna(p-1,j-1))
    
def Mperppj(model, y, p, j):
    Mna = lambda p,j: non_adiab_moment_mapping(model, y, p, j)

    tau = model.p.tau
    q = 1.0
    cpp2 = np.sqrt((p+1)*(p+2))
    cp   = 2*p + 1
    cpm2 = np.sqrt(p*(p-1))
    cjp1 = (2*j + 1)
    cj   = -(j+1)
    cjm1 = -j
            
    return tau/q * (
        model.Cperp(cpp2 * Mna(p+2,j) + cp * Mna(p,j) + cpm2 * Mna(p-2,j))
        + model.Cperp(cjp1 * Mna(p,j+1) + cj * Mna(p,j) + cjm1 * Mna(p,j-1))
    )
    
def Dpj(model, y, p,j):
    if p == 0:
        return - model.p.RN * kernel(model,j) * model.iky * y[-1] \
            - model.p.RT * \
                (2 * j * kernel(model,j) - (j+1) * kernel(model,j+1) - j * kernel(model,j-1)) \
                    * model.iky * y[-1]
    elif p == 2:
        return - model.p.RT * 0.5 * np.sqrt(2) * kernel(model,j) * model.iky * y[-1]
    else:
        return model.zeros
    
def GMX(model, t, y):
    """
    Compute the right-hand side of the fluid equations using the 9GM framework.
    """

    # Access attributes from the model instance
    y = model.poisson_solver.solve(y)
    Mpar = lambda p,j: Mparapj(model, y, p, j)
    Mperp = lambda p,j: Mperppj(model, y, p, j)
    Dia = lambda p,j: Dpj(model, y, p, j)

    for n in range(9):
        p, j = ntopj(n)
        model.dydt[n] = Mpar(p, j) + Mperp(p, j) + Dia(p, j)

    return model.dydt