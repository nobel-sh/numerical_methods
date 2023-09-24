import sympy as sp
from typing import Callable,List
import matplotlib.pyplot as plt
import numpy as np
import random

def fixed_point_iteration(g: Callable[[float], float], initial_guess: float, N: int = 20, max_iterations: int = 200) -> [float]:
    """
    Perform fixed-point iteration to find the fixed point of the function g(x).
    
    Args:
        g (Callable[[float], float]): The function for which the fixed point is sought.
        initial_guess (float): The initial approximation for the fixed point.
        N (int): Number of decimal places for accuracy.
        max_iterations (int): Maximum number of iterations.

    Returns:
        List[float]: List of approximate fixed points during the iterations.

    Fixed-point iteration finds the fixed point of a function g(x) by repeatedly
    applying it to an initial guess until convergence. The function returns a list
    of approximate fixed points, including the initial guess.

    Example:
        >>> g = lambda x: x**2 + 2
        >>> initial_guess = 1.0
        >>> results = fixed_point_iteration(g, initial_guess, N=6, max_iterations=100)
        >>> print("Fixed points:", [f"{x:.6f}" for x in results])
    """
    tolerance: float = 0.5 * 10 ** (-N)
    current_approximation: float = initial_guess
    approximation_list = [current_approximation]
    for iteration in range(max_iterations):
        next_approximation: float = g(current_approximation)
        approximation_list.append(next_approximation)
        if abs(next_approximation - current_approximation) < tolerance:
            return approximation_list
        current_approximation = next_approximation
    return approximation_list

def print_approximation_list(approximation_list: [float], N: int = 20):
    """
    Print the list of approximations obtained during fixed-point iteration.

    Args:
        approximation_list (List[float]): List of approximate values obtained during iterations.
        N (int): Number of decimal places for formatting.

    Returns:
        None

    This function iterates through the list of approximate values obtained during
    fixed-point iteration and prints each value along with the absolute difference
    from the next approximation. It is useful for displaying the progression of
    approximations over iterations.

    Example:
        >>> results = [1.000000, 1.500000, 1.416667, 1.414216]
        >>> print_approximation_list(results, N=6)
    """
    for index in range(len(approximation_list) - 1):
        print(f"Iteration {index} :")
        print(f"\tcurrent approximation =  {approximation_list[index]:.{N}f}")
        print(f"\tnext approximation    =  {approximation_list[index+1]:.{N}f}")
        print(f"\tabsolute difference   =  {abs(approximation_list[index+1] - approximation_list[index]):.{N}f}")


def is_convergent(g: sp.Expr, x0: float) -> bool:
    """
    Check if a function g(x) is convergent at a given point x0.

    Args:
        g (sp.Expr): The symbolic expression representing the function g(x).
        x0 (float): The point at which convergence is checked.

    Returns:
        bool: True if g(x) is convergent at x0, False otherwise.

    This function determines the convergence of a given function g(x) at a specific point x0
    by calculating the absolute value of its derivative at that point. If the absolute value
    of the derivative is less than 1, it is considered convergent, indicating that the fixed-point
    iteration for g(x) at x0 is likely to converge.

    Example:
        >>> g_x = sp.sin(sp.symbols('x')) / 2
        >>> x0 = 1.0
        >>> is_convergent(g_x, x0)
        True
    """
    symbolic_variable: sp.Symbol = sp.symbols('x')
    derivative_g: sp.Expr = sp.diff(g, symbolic_variable)
    absolute_derivative_at_x0: float = abs(derivative_g.subs(symbolic_variable, x0))
    return absolute_derivative_at_x0 < 1

def generate_random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def plot_iteration_convergence_with_function(
        approximation_list: List[float], 
        original_function: Callable[[float], float], 
        x_range: List[float], 
        N: int = 20):
    """
    Create a plot to visualize the convergence of fixed-point iteration along with the original function.

    Args:
        approximation_list (List[float]): List of approximate values obtained during iterations.
        original_function (Callable[[float], float]): The original function g(x).
        x_range (List[float]): Range of x-values for plotting the original function.
        N (int): Number of decimal places for formatting.

    Returns:
        None

    This function generates a plot to visualize the convergence of fixed-point iteration
    and overlays the plot with the original function. It also displays the absolute difference
    between consecutive approximations over iterations.

    Example:
        >>> results = [1.000000, 1.500000, 1.416667, 1.414216]
        >>> g_x = lambda x: x**2 + 2
        >>> x_range = np.linspace(1.0, 2.0, 100)
        >>> plot_iteration_convergence_with_function(results, g_x, x_range, N=6)
    """
    iteration_numbers = range(len(approximation_list) - 1)
    absolute_differences = [abs(approximation_list[i + 1] - approximation_list[i]) for i in iteration_numbers]

    plt.figure(figsize=(10, 6))
    
    # Plot both the original function and the approximations with random colors
    original_color = generate_random_color()
    plt.plot(x_range, [original_function(x) for x in x_range], label='Original Function', linestyle='-', color=original_color)

    # Generate a continuous x-axis based on iteration number
    x_continuous = np.linspace(0, len(approximation_list) - 2, len(approximation_list) - 1)

    for i in range(len(approximation_list) - 1):
        color = generate_random_color()
        plt.scatter(x_continuous[i], approximation_list[i], marker='.', color=color, s=60)
    
    plt.plot(x_continuous, approximation_list[:-1], linestyle='-', color='gray', alpha=0.5,label='Approximations')

    plt.title('Root Finding Using Fixed-Point Iteration')
    plt.legend(loc='lower right')
    plt.xlabel('Iteration / x')
    plt.ylabel('g(x) / Approximations')

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    try:
        f_x: str = input("Enter an equation f(x) or type 'exit' to quit: ")
        # f_x = "sin(x)-10*x+10"
        if f_x.lower() == 'exit':
            exit()
        parsed_f_x: sp.Expr = sp.sympify(f_x)

        g_x: str = input("Provide an equation for g(x) such that x = g(x) from the previously given f(x): ")
        # g_x = "1+sin(x)/10"
        parsed_g_x: sp.Expr = sp.sympify(g_x)
        initial_guess_x: float = float(input("Enter an initial guess: "))

        if not is_convergent(parsed_g_x, initial_guess_x):
            raise Exception("The given function does not converge")

        N: int = int(input("Enter the number of decimal places for accuracy: "))

        max_iterations: int = int(input("Enter the maximum number of iterations: "))

        results: [float] = fixed_point_iteration(lambda x: parsed_g_x.subs('x', x), initial_guess_x, N=N, max_iterations=max_iterations)
        print_approximation_list(results)
        print(f"Fixed point: {results.pop():.{N}f}")
        x_range = np.linspace(1.0, 2.0, 100)
        plot_iteration_convergence_with_function(results, lambda x:parsed_f_x.subs('x',x), x_range, N=20)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()    

# Equations tested:
# ex 2.13 [Introductory Methods of Numerical Analysis, 5th Edition, SS Sastry]
# f(x) = sin(x)-10*x+10
# x = 1+sin(x)/10
# x0 = 1

# ex 2.11 [Introductory Methods of Numerical Analysis, 5th Edition, SS Sastry]
# f(x) = 2x-3-cos(x)
# x = (cos(x)+3)/2
# x0 = 1.75
