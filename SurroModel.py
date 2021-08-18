import numpy as np
from scipy import linalg
from smt.utils import compute_rms_error

from smt.problems import Sphere, NdimRobotArm
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS

try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB

    compiled_available = True
except:
    compiled_available = False

from smt.applications.mixed_integer import MixedIntegerContext, FLOAT, INT, ENUM

xtypes1 = [
    (ENUM, 2),
    (ENUM, 7),
    (ENUM, 5),
    (ENUM, 3),
    (ENUM, 8),
    INT,
    INT,
    # FLOAT,
    # FLOAT,
    # FLOAT,
    # (ENUM, 2)
    (ENUM, 2)
]
xlimits1 = [
    ["False", "True"],
    ["linear", "gen_linear", "cos", "const", "gcn", "gat", "sym-gat"],
    ["mean", "sum", "pool_mean", "pool_max", "mlp"],
    ["mlp", "identity", "none"],
    ["linear", "elu", "sigmoid", "tanh", "relu", "relu6", "softplus", "leaky_relu"],
    [0, 5],
    [0, 5],
    # [0.05, 0.6],
    # [0.05, 0.6],
    # [0.01, 0.49],
    # ["False", "True"]
    ["False", "True"]
]
xtypes2 = [
    (ENUM, 2)
]
xlimits2 = [
    ["False", "True"]
]


###########################################################################
def get_surrogate_model(name, xt, yt, nlayers, ndim, xlimits=None):
    if name == 'LS':
        t = LS(print_prediction=False)
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'QP':
        t = QP(print_prediction=False)
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'KPLS_squar':
        # The variables 'name' must be equal to 'KPLS'. 'n_comp' and 'theta0' must be
        # an integer in [1, ndim[ and a list of length n_comp, respectively. Here is an
        # an example using 2 principal components.
        t = KPLS(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False)
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'KPLS_abs':
        # KPLS + absolute exponential correlation kernel
        # The variables 'name' must be equal to 'KPLS'. 'n_comp' and 'theta0' must be
        # an integer in [1,ndim[ and a list of length n_comp, respectively. Here is an
        # an example using 2 principal components.
        t = KPLS(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False, corr="abs_exp")
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'KRG':
        # The variable 'theta0' is a list of length ndim.
        t = KRG(theta0=[1e-2] * ndim, print_prediction=False)
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'KRG_MIXINT':
        xtps = xtypes1 * nlayers + xtypes2
        xlms = xlimits1 * nlayers + xlimits2
        mixint = MixedIntegerContext(xtps, xlms)
        t = mixint.build_surrogate_model(KRG())
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'KPLSK':
        # 'n_comp' and 'theta0' must be an integer in [1, ndim[ and a list of length n_comp, respectively.
        t = KPLSK(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False)
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'GEKPLS':
        # 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.
        t = GEKPLS(
            n_comp=1,
            theta0=[1e-2],
            xlimits=xlimits,
            delta_x=1e-2,
            extra_points=1,
            print_prediction=False,
        )
        t.set_training_values(xt, yt[0, :])
        # Add the gradient information
        for i in range(ndim):
            t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
        t.train()
    elif name == 'GEKPLS2':
        # 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.
        t = GEKPLS(
            n_comp=1,
            theta0=[1e-2],
            xlimits=xlimits,
            delta_x=1e-4,
            extra_points=2,
            print_prediction=False,
        )
        t.set_training_values(xt, yt[:, 0])
        # Add the gradient information
        for i in range(ndim):
            t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
        t.train()
    elif name == 'IDW':
        t = IDW(print_prediction=False)
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'RBF':
        t = RBF(print_prediction=False, poly_degree=0)
        t.set_training_values(xt, yt)
        t.train()
    elif name == 'RMTC':
        t = RMTC(
            xlimits=xlimits,
            min_energy=True,
            nonlinear_maxiter=20,
            print_prediction=False,
        )
        t.set_training_values(xt, yt[:, 0])
        # Add the gradient information
        for i in range(ndim):
            t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
        t.train()
    elif name == 'RMTB':
        t = RMTB(
            xlimits=xlimits,
            min_energy=True,
            nonlinear_maxiter=20,
            print_prediction=False,
        )
        t.set_training_values(xt, yt[:, 0])
        # Add the gradient information
        for i in range(ndim):
            t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
        t.train()
    else:
        raise RuntimeError('Unknown surrogate.')

    return t


if __name__ == '__main__':
    xtps = xtypes1 * 2  # + xtypes2
    xlms = xlimits1 * 2  # + xlimits2
    mixint = MixedIntegerContext(xtps, xlms)
    t = mixint.build_surrogate_model(KRG())
    print(t)
    from search_space import MacroSearchSpace

    search_space = MacroSearchSpace(tag_all=True)
    xt = np.concatenate([search_space.generate_action_solution_4_surrogate() for _ in range(100)])
    yt = np.random.rand(100, 1)
    t.set_training_values(xt, yt)
    t.train()
