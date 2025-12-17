"""
    A collection of descent methods based on gradients for optimization
    
    @author: Rodolphe Le Riche, Brian Dédji Whannou
"""

import numpy as np
from typing import Callable

from optim_utilities import record_best
from optim_utilities import record_any

################## finite difference function
def gradient_finite_diff(
    func: Callable, x: np.ndarray, f_x: float, epsilon: float = 1e-7
) -> np.ndarray:
    """
    Gradient estimation by forward finite difference

    Costs len(x) calls to func

    Parameters
    ----------
    func : object
        function.
    x : np.ndarray
        point where the gradient is approximated.
    f_x : float
            function value at x.
    epsilon : float, optional
        size of the forward perturbation. The default is 1e-7.

    Returns
    -------
    gradient : 1D np.ndarray
        estimate of the 1D gradient of the partial derivatives at x.

    """
    dimension = len(x)
    gradient = np.zeros(dimension)
    for coordinate_index in range(dimension):
        h = np.zeros(dimension)
        h[coordinate_index] = epsilon
        f_xh = func(x + h)
        gradient[coordinate_index] = (f_xh - f_x) / epsilon

    return gradient


################## line search
def linesearch(
    x: np.ndarray,
    f_x: float,
    gradf: np.ndarray,
    direction: np.ndarray,
    func: Callable,
    LB: np.ndarray,
    UB: np.ndarray,
    rec: dict,
    printlevel: int,
    suffDecFact=0.1,
    decFact=0.5,
    initStepFact: float=1,
):
    """
    line search function

    Backtracking with Armijo (sufficient decrease) condition
    find stepSize that satisfies :
        f(x+stepSize*direction) <= f(x) + stepSize* (suffDecFact * direction^T*gradf)
    If direction is not a descent direction ( direction^T*gradf>0 ),
        turn around (multiply direction by -1).
    In addition we return the stepSize that corresponds to the best function
        tried during the line search which improves the global search
        capacities of the optimizer.

    Parameters
    ----------
    x : np.ndarray
        current point.
    f_x : float
        objective function at x.
    gradf : np.ndarray
        gradient of f at x.
    direction : np.ndarray
        proposed search direction. Not necessarily a descent direction
        (hence works with momentum and NAG).
    func : object
        pointer to objective function.
    LB : List
        lower bounds on x.
    UB : List
        upper bounds on x.
    rec : dict
        recording of points tried.
    printlevel : int
        recording level, =0,1 or more (cf general documentation).
    suffDecFact : TYPE, optional
        sufficient decrease factor, in [0,1[. The default is 0.1.
    decFact : TYPE, optional
        step size decrease factor, in ]0,1[. The default is 0.5.
    initStepFact : TYPE, optional
        initial step factor, gets multiplied by gradient norm to determine
        initial step size. The default is 1.

    Raises
    ------
    ValueError
        some input checking.

    Returns
    -------
    next_x : np.ndarray
        next point found.
    n_loop-1
        cost of line search.
    rec : dict
        updated dictionary of records.

    """

    if not 0 <= suffDecFact < 1:
        raise ValueError(
            "the sufficient decrease factor (suffDecFact) should be between 0 and (strictly) 1"
        )

    normGrad = float(np.linalg.norm(gradf))
    size_of_domain = float(np.linalg.norm(np.array([UB]) - np.array([LB])))
    # calculate initial stepSize
    # either as initStepFact*norm of gradient (but this may fail in flat regions)
    # or as a fraction of domain diagonal. Take the max of both initial step sizes.
    stepSize = max(initStepFact * normGrad, (size_of_domain / 100))
    gradient_projected_on_direction = direction.dot(gradf)
    # if direction is not a descent direction, -direction is, turn around
    if gradient_projected_on_direction > 0:
        direction = -direction
        gradient_projected_on_direction = -gradient_projected_on_direction

    n_loop = 0
    maxloop = 100  # max line search budget

    f_ls_best = float("inf")
    x_ls_best = x * np.nan

    condition = False

    while not condition:
        next_x = x + stepSize * direction
        # coordinates should be inside the domain
        next_x_inbounds = np.where(next_x < LB, LB, np.where(next_x > UB, UB, next_x))
        # only evaluate next_x if it is in-bounds, otherwise decrease step size
        if np.linalg.norm(next_x_inbounds - next_x) < 10e-10:
            f_next = func(next_x_inbounds)
            n_loop += 1
            rec = record_any(
                rec=rec,
                f=f_next,
                x=next_x_inbounds,
                time=(rec["time_used"] + 1),
                printlevel=printlevel,
            )
            if f_next < f_ls_best:  # record best of linesearch
                x_ls_best = next_x_inbounds
                f_ls_best = f_next
        else:
            f_next = float("inf")

        condition_loop = n_loop >= maxloop
        condition_decrease = f_next < (
            f_x + suffDecFact * stepSize * gradient_projected_on_direction
        )
        condition = condition_loop or condition_decrease
        stepSize = decFact * stepSize

    # return global best of line search : this part makes the optimizer
    # global. Remove it to return to a classical local search (that gets
    # trapped in local optima)
    if f_ls_best < f_next:
        next_x_inbounds = x_ls_best
        f_next = f_ls_best

#    return next_x_inbounds, (n_loop - 1), rec
    return next_x_inbounds, f_next, n_loop, rec


############### gradient based searches
def gradient_descent_patched(
    func: Callable,
    start_x: np.ndarray,
    LB: np.ndarray,
    UB: np.ndarray,
    budget: int = 1e3,
    step_factor: float = 1e-1,
    direction_type: str = "momentum",
    do_linesearch: bool = True,
    min_step_size: float = 1e-11,
    min_grad_size: float = 1e-6,
    inertia: float = 0.9,
    printlevel: int = 1,

    step_decay_type: str = "none",
    step_decay_rate: float = 1e-3

) -> dict:
    """
        A collection of descent algorithms that use gradients
        The algorithms differ in how the search direction is calculated
            direction_type = (gradient, momentum, NAG) , where NAG= Nesterov Accelerated Gradient
        and they differ on whether a single step is taken or a line search performed.


    Parameters
    ----------
    func : Callable
        function to minimize
    start_x : np.ndarray
        starting point
    LB, UB : np.ndarray
        lower and upper bounds for x, dimension of x
    budget: int
        maximum number of calls to func (default= 1e3)
    step_factor: float
        multiplies -gradient of func to define the basic (gradient) step (default= 1e-1)
    direction_type: str
        = "momentum", "gradient", "NAG"
    do_linesearch: bool
        default = True
    min_step_size: float
        when (norm step size)/sqrt(dim) below it, algo stops (default = 1e-11)
    min_grad_size: float
        when (norm of gradient)/sqrt(dim) below it, algo stops (default = 1e-6)
    inertia: float
        inertia factor in momentum and NAG, default = 0.9
    printlevel: int
        controls how much is stored and printed
        =0 store overall best only, no plot
        =1 store best history, do plots
        =2 store best history and all points, do plots


    Returns
    -------
    res : dict
        a dictionary with fields that depend on printlevel.
        res["time_used"] : how many evaluations of the objective function have been done
        res["x_best"], res["f_best"] : x and f of best, always recorded
        res["stop_condition"] : why algo stopped
        res["hist_x_best"], res["hist_f_best"], res["hist_time_best"] : record of x,f,time found for new best so far
        res["hist_x"], res["hist_f"], res["hist_time"] : record of x,f,time for all points sampled

    Raises
    ------
    ValueError
        when input not consistent
    """

    if len(LB) != len(start_x) or len(UB) != len(start_x):
        raise ValueError(
            "inconsistent size of LB, %s, UB, %s, and start_x, %s" % len(LB),
            len(UB),
            len(start_x),
        )

    # initializations
    dim = len(start_x)
    iteration = 0
    nb_fun_calls = 0
    best_f = float("inf")
    best_x = start_x
    current_x = start_x
    res = {}  # results dictionary
    condition = False
    previous_step = np.zeros(dim)

    # start search
    while not condition:
        # calculate f and its gradient
        current_f = func(current_x)
        nb_fun_calls += 1
        current_gradient = gradient_finite_diff(func, current_x, current_f)
        nb_fun_calls += dim  # cost of the forward finite difference such as implemented
        # book-keeping
        iteration += 1
        res = record_any(
            rec=res, f=current_f, x=current_x, time=nb_fun_calls, printlevel=printlevel
        )
        if current_f < best_f:
            best_x = current_x
            best_f = current_f
            res = record_best(
                rec=res,
                fbest=best_f,
                xbest=best_x,
                time=nb_fun_calls,
                printlevel=printlevel,
            )
        previous_x = current_x

        # determine search direction
        gradient_size = np.linalg.norm(current_gradient)
        condition_gradient = (
            np.linalg.norm(gradient_size) / np.sqrt(dim)
        ) <= min_grad_size
        # it does not make sense to do the rest if at a null-gradient point and
        # there is a risk of exception error
        if not condition_gradient:

            if direction_type == "gradient":
                # pas potentiellement décroissant
                if step_decay_type == "linear":
                    current_step_factor = step_factor * (1.0 - iteration / budget)
                elif step_decay_type == "exponential":
                    current_step_factor = step_factor * (step_decay_rate ** iteration)
                elif step_decay_type == "inverse_time":
                    current_step_factor = step_factor / (1.0 + step_decay_rate * iteration)
                elif step_decay_type == "step_decay":
                    current_step_factor = step_factor * (0.5 ** (iteration // 100))
                else:
                    current_step_factor = step_factor

                delta_x = -current_step_factor * current_gradient
            elif direction_type == "momentum":
                # pas potentiellement décroissant
                if step_decay_type == "linear":
                    current_step_factor = step_factor * (1.0 - iteration / budget)
                elif step_decay_type == "exponential":
                    current_step_factor = step_factor * (step_decay_rate ** iteration)
                elif step_decay_type == "inverse_time":
                    current_step_factor = step_factor / (1.0 + step_decay_rate * iteration)
                elif step_decay_type == "step_decay":
                    current_step_factor = step_factor * (0.5 ** (iteration // 100))
                else:
                    current_step_factor = step_factor

                if iteration <= 1:
                    delta_x = -current_step_factor * current_gradient
                else:
                    delta_x = -current_step_factor * current_gradient + inertia * previous_step
            elif direction_type == "NAG":
                # pas potentiellement décroissant
                if step_decay_type == "linear":
                    current_step_factor = step_factor * (1.0 - iteration / budget)
                elif step_decay_type == "exponential":
                    current_step_factor = step_factor * (step_decay_rate ** iteration)
                elif step_decay_type == "inverse_time":
                    current_step_factor = step_factor / (1.0 + step_decay_rate * iteration)
                elif step_decay_type == "step_decay":
                    current_step_factor = step_factor * (0.5 ** (iteration // 100))
                else:
                    current_step_factor = step_factor

                if iteration <= 1:
                    delta_x = -current_step_factor * current_gradient
                else:
                    xnag = current_x + inertia*previous_step
                    fnag = func(xnag)
                    nb_fun_calls += 1
                    gradnag = gradient_finite_diff(func, xnag, fnag) # finite difference will make NAG expensive
                    nb_fun_calls += dim  # cost of the forward finite difference such as implemented
                    delta_x = -current_step_factor * gradnag + inertia * previous_step
            else:
                raise ValueError("unknown direction_type " + direction_type)

            # if the current point is near a boundary, the direction should be projected on that boundary
            tol = 1.0e-15
            violation = np.where(current_x > (np.array(LB) + tol), 0, -1) + np.where(
                current_x < (np.array(UB) - tol), 0, 1
            )
            delta_x[np.where(violation * delta_x > 0)] = 0

            if np.linalg.norm(delta_x) < tol:
                condition_step = True
            else:
                direction = delta_x / np.linalg.norm(delta_x)
                condition_step = False

            # step in direction, with or without line search, to get new x
            if do_linesearch and not condition_step:
                current_x, current_f, linesearch_cost, res = linesearch(
                    x=current_x,
                    f_x=current_f,
                    gradf=current_gradient,
                    direction=direction,
                    func=func,
                    LB=LB,
                    UB=UB,
                    rec=res,
                    printlevel=printlevel,
                )
                nb_fun_calls += linesearch_cost
                delta_x = current_x - previous_x
            else:
                current_x = previous_x + delta_x
                # project point in-bounds
                current_x = np.where(
                    current_x < LB, LB, np.where(current_x > UB, UB, current_x)
                )

            # not sure which of these 2 versions is best, usually equivalent
#            previous_step = delta_x # version marginally corrected by projection on bounds
            previous_step = current_x - previous_x # version corrected by linesearch and projection on bounds

        # check stopping conditions
        condition_iteration = nb_fun_calls >= budget
        condition_step = (np.linalg.norm(current_x - previous_x)/ np.sqrt(dim)) <= min_step_size
        condition = condition_iteration or condition_step or condition_gradient

    stop_condition = str()
    if condition_iteration:
        stop_condition += "budget exhausted "
    if condition_step:
        stop_condition += "too small step "
    if condition_gradient:
        stop_condition += "too small gradient"
    res["stop_condition"] = stop_condition
    return res


###########################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from typing import Callable, List
    from test_functions import (
        linear_function,
        ackley,
        sphere,
        quadratic,
        rosen,
        L1norm,
        sphereL1
    )
    from optim_utilities import print_rec
    from random_search import random_opt

    # function definition
    fun = rosen
    dim = 2
    LB = np.array([-5] * dim)
    UB = np.array([5] * dim)
    np.random.seed(123) # useful for repeated runs (quadratic fct or initial random point)

    #########################
    # algorithms settings
    # start_x = np.array([3,2,1,-4.5,4.6,-2,-1,4.9,0,2])
    # start_x = (1+np.arange(dim))*5/dim
    # start_x = np.array([2.3,4.5])
    start_x = np.random.uniform(low=LB, high=UB)

    budget = 1000 * (dim + 1)
    printlevel = 1  # =0,1,2 , careful with 2 which is memory consuming

    #########################
    # optimize
    # res = random_opt(func=fun, LB=LB, UB=UB, budget=budget, printlevel=printlevel)
    res = gradient_descent_patched(func=fun, start_x=start_x, LB=LB, UB=UB, budget=budget,
                           step_factor=0.0001, direction_type="momentum",
                           do_linesearch=False, min_step_size=1e-11,
                           min_grad_size=1e-6, inertia=0.9, printlevel=printlevel,
                           step_decay_type="none", step_decay_rate=1e-3)

    #########################
    # reporting
    print_rec(res=res, fun=fun, dim=dim, LB=LB, UB=UB, printlevel=printlevel, logscale=True)

