from dataclasses import dataclass
import numpy as np
import time


@dataclass
class QuadraticCostStruct:
    Q: np.ndarray
    g: np.ndarray

    def __post_init__(self):
        if not isinstance(self.Q, np.ndarray):
            raise TypeError("Q must be a numpy ndarray")
        if not isinstance(self.g, np.ndarray):
            raise TypeError("g must be a numpy ndarray")
        if not np.allclose(self.Q, self.Q.T):
            raise ValueError("Q must be symmetric to check PSD")
        # eigvals = np.linalg.eigvalsh(self.Q)
        # if np.any(eigvals < -1e-8):
            # raise ValueError("Q must be positive semi-definite (PSD)")
        if self.g.shape[0] != self.Q.shape[0]:
            raise ValueError("Q and g must have the same number of rows")

    def __str__(self):
        return f"QuadraticCostStruct(\nQ=\n{self.Q}\ng=\n{self.g}\n)\n"
    
    def unconstrained_optimal(self):
        return -np.linalg.solve(self.Q, self.g)

    def evaluate_cost(self, y: np.ndarray):
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray")
        if y.shape[0] != self.Q.shape[0]:
            raise ValueError("y must have the same number of elements as Q rows")
        return 0.5 * y.T @ self.Q @ y + self.g.T @ y

    def EigenDecomp(self):
        eigvals, eigvecs = np.linalg.eigh(self.Q)
        return eigvals, eigvecs



@dataclass
class SolverInput:
    cost_struct: QuadraticCostStruct
    radius: float

    tolerance: float = 1e-6

    use_eigendecomp: bool = True  # Set to False if you want to use the non-eigendecomp version

    def __post_init__(self):
        if not isinstance(self.cost_struct, QuadraticCostStruct):
            raise TypeError("cost_struct must be an instance of QuadraticCostStruct")
        if not isinstance(self.radius, (int, float)):
            raise TypeError("radius must be a numeric type")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        
    
    def __str__(self):
        return f"SolverInput(\ncost_struct=\n{self.cost_struct}\nradius={self.radius}\n)\n"
    


class PrecomputedSolverData:
    V : np.ndarray = None
    eig : np.ndarray = None
    v_tsps_g: np.ndarray = None
    v_tsps_g_sq: np.ndarray = None
    vg_over_dsq: float = 0.0

    def __init__(self, solver_input: SolverInput):
        if not isinstance(solver_input, SolverInput):
            raise TypeError("solver_input must be an instance of SolverInput")
        self.solver_input = solver_input
        self.eig, self.V = self.solver_input.cost_struct.EigenDecomp()
        self.v_tsps_g = self.V.T @ self.solver_input.cost_struct.g
        self.v_tsps_g_sq = self.v_tsps_g ** 2
        self.vg_over_dsq = (self.v_tsps_g / self.solver_input.radius) ** 2


class TRP_solver_base:
    """
    Base class for TRP solvers.
    """
    solver_input: SolverInput
    lambda_: float = 0.0
    use_eigendecomp: bool = True  # Set to False if you want to use the non-eigendecomp version

    precomputed_data: PrecomputedSolverData = None
    
    def __init__(self, solver_input: SolverInput):
        if not isinstance(solver_input, SolverInput):
            raise TypeError("solver_input must be an instance of SolverInput")
        self.solver_input = solver_input
        self.use_eigendecomp = solver_input.use_eigendecomp
        if self.use_eigendecomp:
            self.precomputed_data = PrecomputedSolverData(solver_input)
        else:
            self.precomputed_data = None

        self.lambda_ = 0.0
    
       
    def set_lambda(self, lambda_value: float):
        if not isinstance(lambda_value, (int, float)):
            raise TypeError("lambda_value must be a numeric type")
        if lambda_value < 0:
            raise ValueError("lambda_value must be non-negative")
        self.lambda_ = lambda_value


    def set_initial_guess(self, initial_guess: np.ndarray):
        r_guess = np.linalg.norm(initial_guess)
        nabla_f_guess = self.solver_input.cost_struct.Q @ initial_guess + self.solver_input.cost_struct.g
        self.lambda_ = (np.linalg.norm(nabla_f_guess) / r_guess)


    def set_default_initial_guess(self):
        # self.set_lambda(0.01)
        self.lambda_ = np.linalg.norm(self.solver_input.cost_struct.g) / self.solver_input.radius
        # unconstrained_optimal = self.solver_input.cost_struct.unconstrained_optimal()
        # guess = self.solver_input.radius * (unconstrained_optimal / np.linalg.norm(unconstrained_optimal))
        # self.set_initial_guess(guess)
        
    def step_basic(self):
        """ This method should be implemented in subclasses to define the stepping strategy.
        It should return a convergence metric.
        """
        raise NotImplementedError("step_basic() must be implemented in a subclass.")
    
    def step_w_eigendecomp(self):
        """
        This method should be implemented in subclasses to define the stepping strategy using eigendecomposition.
        It should return a convergence metric.
        """
        raise NotImplementedError("step_w_eigendecomp() must be implemented in a subclass.")

    def step(self): 
        if(self.use_eigendecomp):
            return self.step_w_eigendecomp()
        else:
            return self.step_basic()

    def recover_incremental_solution(self):
        if not(self.use_eigendecomp):
            Q_plus_lambda = self.solver_input.cost_struct.Q + self.lambda_ * np.eye(self.solver_input.cost_struct.Q.shape[0])
            return np.linalg.solve(Q_plus_lambda, -self.solver_input.cost_struct.g)
        else:
            eig_plus_lambda_inv = 1.0 / (self.precomputed_data.eig + self.lambda_)
            return - self.precomputed_data.V @ (eig_plus_lambda_inv * self.precomputed_data.v_tsps_g)

    def solve(self, iteration_limit=1e6):
        # The following is an appropriate upper bound for the dual variable lambda
        # however, we will not use it in the current implementation. Instead, we expect the initial guess to be set beforehand.
        # if(self.use_eigendecomp):
        #     self.set_lambda( np.sum(self.precomputed_data.vg_over_dsq) )
        iterations = 0
        while True:
            convergence_metric = self.step()
            #The "convergence_metric" can be any metric, defined in the "step" function.
            # It may be a good idea to set it to be the difference between the current and desired solution norm, however, this requires recovering the solution norm at each step.
            # In the case that that takes too long, the convergence_metric could be the difference between the current and previous dual variable (lambda).
            iterations += 1
            if convergence_metric < self.solver_input.tolerance:
                break
            if iterations >= iteration_limit:
                print(f"Iteration limit reached: {iteration_limit}. Stopping the solver.")
                break
        return self.recover_incremental_solution(), iterations

    def solve_with_experiment_result_data(self, iteration_limit=1e6):
        # if(self.use_eigendecomp):
        #     self.set_lambda(np.sum(self.precomputed_data.vg_over_dsq))

        iteration_number = 0
        total_time = 0.0
        experiment_result_data = {}
        experiment_result_data[0] = dict()
        experiment_result_data[0]['lambda'] = self.lambda_
        experiment_result_data[0]['convergence_metric'] = 0  # Initial convergence metric is None
        y_incr = self.recover_incremental_solution()
        experiment_result_data[0]['incremental_solution'] = y_incr
        experiment_result_data[0]['radius'] = np.linalg.norm(y_incr)
        experiment_result_data[0]['step_time'] = 0.0  # Initial step time is 0
        experiment_result_data[0]['cost'] = self.solver_input.cost_struct.evaluate_cost(y_incr / np.linalg.norm(y_incr))
        while True:
            start_time = time.time()
            convergence_metric = self.step()
            step_time = time.time() - start_time
            iteration_number += 1
            y_incr = self.recover_incremental_solution()
            experiment_result_data[iteration_number] = dict()
            experiment_result_data[iteration_number]['lambda'] = self.lambda_
            experiment_result_data[iteration_number]['convergence_metric'] = convergence_metric
            experiment_result_data[iteration_number]['incremental_solution'] = y_incr
            experiment_result_data[iteration_number]['radius'] = np.linalg.norm(experiment_result_data[iteration_number]['incremental_solution'])
            experiment_result_data[iteration_number]['step_time'] = step_time
            experiment_result_data[iteration_number]['cost'] = self.solver_input.cost_struct.evaluate_cost(y_incr / np.linalg.norm(y_incr))
            total_time += step_time

            if convergence_metric < self.solver_input.tolerance:
                break
            if iteration_number >= iteration_limit:
                print(f"Iteration limit reached: {iteration_limit}. Stopping the solver.")
                break


        return self.recover_incremental_solution(), iteration_number, experiment_result_data, total_time

    def Q(self):
        """
        Returns the cost matrix Q from the solver input.
        """
        return self.solver_input.cost_struct.Q
    def radius(self):
        """
        Returns the radius from the solver input.
        """
        return self.solver_input.radius
    def g(self):
        """
        Returns the gradient vector g from the cost structure.
        """
        return self.solver_input.cost_struct.g

