import numpy as np
import pandas as pd
import scipy.stats
from quantum_solvers_with_deflation import classical_hhl_large_system_with_deflation

from quantum_solvers import quantum_hhl_large_system, hybrid_hhl_solution
from scipy.linalg import eigh, solve

from withDeflation import quantum_hhl_large_system_with_deflation

# Parameters
n = np.pow(2,8) - 2   # Number of discretization points without the boundary nodes
print(n)
L = 1.0  # Length of the domain
h = L / (n + 1)  # Spatial step size
delta_s = 1e-3  # Initial step size for arc-length continuation
min_delta_s = 1e-3  # Minimum step size quantum_hhl_large_system
max_delta_s = 0.1  # Maximum step size
tol = 1e-5  # Convergence tolerance
x = np.linspace(0, 1, n + 2)  # Include boundary nodes
newtonSteps = 1500

def F(u, lambda_param):
    f = np.zeros(n + 2)  # Include boundary points in f
    for i in range(1, n + 1):
        # Finite difference approximation
        f[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / h ** 2 + lambda_param * np.exp(u[i])
    # Boundary conditions
    f[0] = u[0]  # u(0) = 0
    f[-1] = u[-1]  # u(1) = 0
    return f

def jacobian(u, lambda_param):
    jacobian = np.zeros((n + 2, n + 2))  # Include boundary points in Jacobian
    for i in range(1, n + 1):
        jacobian[i, i - 1] = 1.0 / h ** 2
        jacobian[i, i] = -2.0 / h ** 2 + lambda_param * np.exp(u[i])
        jacobian[i, i + 1] = 1.0 / h ** 2

    # Boundary conditions for the Jacobian
    jacobian[0, :] = 0  # u(0) = 0
    jacobian[-1, :] = 0  # u(1) = 0
    jacobian[0, 0] = 1  # u(0) = 0
    jacobian[-1, -1] = 1  # u(1) = 0

    return jacobian

def save_solution_to_excel(filename, solutions, L, n):
    """
    Save solutions to an Excel file.

    Parameters:
    - filename (str): The name of the Excel file to save the solutions.
    - solutions (list of tuples): Each tuple contains (u, lam), where u is the solution array and lam is the parameter.
    - L (float): Length of the domain.
    - n (int): Number of intervals in the domain.

    Returns:
    - None
    """
    x = np.linspace(0, L, n + 2)  # Create the x-values

    # Initialize a dictionary to store the data
    data = {"x": x}

    # Add each solution to the dictionary
    for idx, (u, lam) in enumerate(solutions):
        data[f"u_{idx + 1} (λ={lam:.3f})"] = u

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False)

    print(f"Solutions saved to {filename}")

# Pseudo arc-length continuation solver with augmented Jacobian
def pseudo_arc_length_continuation(u0, lam0, max_steps=1500, save_interval=10):
    global delta_s  # Declare delta_s as global to modify it

    results = []  # To store results for saving to Excel
    norm_and_lambda = []  # To store the norm of delta and lambda

    # First solution
    for _ in range(newtonSteps):
        F_val = F(u0, lam0)
        J = jacobian(u0, lam0)

        # Solve for Newton step
        try:
            #delta = hybrid_hhl_solution(J, -F_val)
            #delta = quantum_hhl_large_system(J, -F_val)
            delta = solve(J, -F_val)
            print ( np.linalg.norm(delta) )

        except np.linalg.LinAlgError:
            #print("Matrix is singular; stopping continuation.")
            return results, norm_and_lambda

        u0 += delta

        if np.linalg.norm(delta) < tol:
            break

    solutions = [(u0, lam0)]

    save_solution_to_excel("0_1_c.xlsx", solutions, L, n)

    u_prev, lam_prev = u0.copy(), lam0
    delta_u = np.ones(n+2) * 1e-3  # Small non-zero initial tangent for u
    delta_lambda = 1e-3  # Small non-zero initial tangent for lambda

    for step in range(max_steps):
        # Predictor step
        delta = 10000

        u_pred = u_prev + delta_s * delta_u
        lam_pred = lam_prev + delta_s * delta_lambda

        step1 = 0
        # Newton-Raphson corrector step
        while np.linalg.norm(delta) > tol:  # Maximum 10 Newton iterations
            F_val = F(u_pred, lam_pred)
            J = jacobian(u_pred, lam_pred)

            # Construct the augmented Jacobian
            F_aug = np.append(F_val, (u_pred - u_prev) @ delta_u + (lam_pred - lam_prev) * delta_lambda - delta_s)

            # Formulate the augmented matrix
            J_aug = np.zeros((n + 3, n + 3))

            J_aug[:n+2, :n+2] = J
            J_aug[:n+2, -1] = np.exp(u_pred)  # Add lambda partial derivatives

            J_aug[-1, :-1] = delta_u
            J_aug[-1, -1] = delta_lambda

            J_aug[0, -1] = 0  # To impose the boundary conditions
            J_aug[-2, -1] = 0  # To impose the boundary conditions

            try:
                #delta = quantum_hhl_large_system(J_aug, -F_aug)
                delta = solve(J_aug, -F_aug)
            except np.linalg.LinAlgError:
                print("Matrix is singular; stopping continuation.")
                return results, norm_and_lambda

            u_pred += delta[:-1]
            lam_pred += delta[-1]

            step1 = step1 + 1

            # Check for convergence
            if np.linalg.norm(delta) < tol or step1 == newtonSteps:
                print(np.linalg.norm(delta), step1)
                # Record the norm of delta and the corresponding lambda
                break

        # Update solution and tangent direction
        solutions.append((u_pred.copy(), lam_pred))

        # Save every `save_interval` steps
        if step % save_interval == 0:
            norm_and_lambda.append({'Step': step, 'Lambda': lam_pred, 'Norm of Delta': np.linalg.norm(delta)})
            results.append({'Step': step, 'Lambda': lam_pred, 'Solution': u_pred.copy()})

        # Dynamic adjustment of delta_s
        delta_s = min(max_delta_s, delta_s * 1.5) if np.linalg.norm(delta) < tol else max(min_delta_s, delta_s * 0.5)

        delta_u = u_pred - u_prev
        delta_lambda = lam_pred - lam_prev
        norm = np.sqrt(delta_u @ delta_u + delta_lambda ** 2)
        delta_u /= norm
        delta_lambda /= norm

        u_prev, lam_prev = u_pred.copy(), lam_pred

        print(f"Step {step}, Lambda: {lam_pred}")

    return results, norm_and_lambda

# Initial setup
u0 = np.zeros(n+2)  # Initial guess for u
lambda0 = 1.0  # Starting lambda

# Run continuation
results, norm_and_lambda = pseudo_arc_length_continuation(u0, lambda0)

with pd.ExcelWriter("arc_length_continuation_results.xlsx", engine="openpyxl") as writer:
    # Save summary in the first sheet
    data = [{'Step': res['Step'], 'Lambda': res['Lambda'], 'Midpoint_u': res['Solution'][n // 2]} for res in results]
    df_summary = pd.DataFrame(data)
    df_summary.to_excel(writer, sheet_name="Summary", index=False)

    # Save norm of delta and lambda in a separate sheet
    df_norms_and_lambdas = pd.DataFrame(norm_and_lambda)
    df_norms_and_lambdas.to_excel(writer, sheet_name="Norms_and_Lambdas", index=False)

    # Save u vs x for each lambda in separate sheets
    for res in results:
        df_solution = pd.DataFrame({
            'x': x,
            'u(x)': res['Solution']
        })
        sheet_name = f"Lambda_{res['Lambda']:.4f}"
        df_solution.to_excel(writer, sheet_name=sheet_name[:31], index=False)

print("Results saved to arc_length_continuation_results.xlsx")