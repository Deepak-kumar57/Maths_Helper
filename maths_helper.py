# Import necessary libraries
import streamlit as st
from sympy import (
    symbols, diff, sympify, solve, latex, lambdify, Eq, SympifyError, Equality,
    simplify, N, plot_implicit, And, Or, Symbol
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import re # For parsing inequalities

# --- Helper Functions ---
def parse_expression_safely(expr_str, local_dict=None, transformations=None):
    """Safely parses a string into a SymPy expression."""
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(expr_str, local_dict=local_dict, transformations=transformations)

# --- Base Class for Operations ---
class MathOperation:
    """Base class for mathematical operations.Each operation should implement display_ui_and_process."""
    def __init__(self):
        # Defining common symbols that can be used in expressions
        self.x, self.y, self.z, self.t = symbols('x y z t')
        self.common_symbols = {'x': self.x, 'y': self.y, 'z': self.z, 't': self.t}
        # Adding common math functions to the local dict for parsing
        self.parse_local_dict = {sym_name: sym_obj for sym_name, sym_obj in self.common_symbols.items()}
        self.parse_local_dict.update({
            "sin": sympify("sin"), "cos": sympify("cos"), "tan": sympify("tan"),
            "exp": sympify("exp"), "log": sympify("log"), "sqrt": sympify("sqrt"),
            "pi": sympify("pi"), "I": sympify("I") # I for imaginary unit
        })


    def display_ui_and_process(self):
        """Placeholder for displaying UI elements and processing logic.This method should be overridden by subclasses."""

    def _plot_(self, ax, title_expr_latex, y_vals_list=None, x_label="$x$", y_label="$y$"):
        """Helper to beautify Matplotlib plots."""
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title_expr_latex)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.7)
        ax.axvline(0, color='black', linewidth=0.7)

        ax.legend()


# --- Derivative Calculator ---
class DerivativeCalculator(MathOperation):
    def display_ui_and_process(self):
        st.header("📈 Derivative Calculator")
        expression_str_deriv = st.text_input(
            "Enter a mathematical expression (e.g., x**2 * sin(y) + exp(x*y)):",
            value="x**2 * sin(y) + exp(x*y)",
            key="deriv_expr"
        )
        variables_input_deriv = st.text_input(
            "Enter variables for differentiation (comma-separated, e.g., x,y):",
            value="x,y",
            key="deriv_vars"
        )

        if st.button("Calculate Derivatives", key="calc_deriv_button"):
            declared_vars_map = {}
            parsed_vars_sym = []

            if variables_input_deriv.strip():
                try:
                    var_names = [var.strip() for var in variables_input_deriv.split(',') if var.strip()]
                    for var_name in var_names:
                        if var_name not in declared_vars_map: # Ensure unique symbols
                           sym = symbols(var_name)
                           declared_vars_map[var_name] = sym
                           parsed_vars_sym.append(sym)
                except (SyntaxError, TypeError) as e:
                    st.error(f"Invalid variable name(s): {e}. Please use valid symbols separated by commas.")
                    return
            
            # Update parse_local_dict with user-declared variables for this specific calculation
            current_parse_local_dict = self.parse_local_dict.copy()
            current_parse_local_dict.update(declared_vars_map)


            if not expression_str_deriv:
                st.warning("Please enter a mathematical expression.")
                return

            try:
                expr_deriv = parse_expression_safely(expression_str_deriv, local_dict=current_parse_local_dict)
                st.write("Original Expression:")
                st.latex(latex(expr_deriv))
                st.write("---")

                for var_sym in parsed_vars_sym:
                    st.markdown(f"### Partial Derivative with respect to ${latex(var_sym)}$:")
                    try:
                        derivative_evaluated = diff(expr_deriv, var_sym).doit()
                        derivative_simplified = simplify(derivative_evaluated)
                        st.latex(latex(derivative_simplified))
                    except (TypeError, ValueError, AttributeError) as diff_err:
                        st.error(f"Could not differentiate with respect to ${latex(var_sym)}$: {diff_err}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during differentiation for variable ${latex(var_sym)}$: {e}")
                    st.write("---")

            except (SympifyError, SyntaxError) as e:
                st.error(f"Error parsing expression: {e}. Ensure variables used in the expression are declared for differentiation or are standard (x, y, z, t).")
            except Exception as e:
                st.error(f"An unexpected error occurred during differentiation setup: {e}")

# --- Single Variable Equation Solver ---
class EquationSolver(MathOperation):
    def display_ui_and_process(self):
        st.header("🔓 Equation Solver (Single Variable)")
        equation_str_solve = st.text_input(
            "Enter an equation in terms of a single variable (e.g., 6*x**2 + 11*x - 6 = 0 or y**2 == 4):",
            value="x**2 - 4 = 0",
            key="solve_eq"
        )

        if st.button("Solve Equation", key="solve_button"):
            if not equation_str_solve:
                st.warning("Please enter an equation to solve.")
                return

            try:
                # Attempt to identify the variable
                temp_expr_for_vars = None
                if '=' in equation_str_solve: # Treat single '=' as '=='
                    lhs_str, rhs_str = equation_str_solve.split('=', 1)
                    temp_expr_for_vars = parse_expression_safely(f"({lhs_str}) - ({rhs_str})", local_dict=self.parse_local_dict)
                
                potential_vars = list(temp_expr_for_vars.free_symbols)

                if len(potential_vars) > 1:
                    st.warning(f"Please use only one variable. Found: {', '.join(map(str, potential_vars))}")
                    return

                var_solve = potential_vars[0]
                st.write(f"Detected variable: ${latex(var_solve)}$")

                # Create a local dict specific to this variable for parsing the actual equation
                current_parse_local_dict = self.parse_local_dict.copy()
                current_parse_local_dict[str(var_solve)] = var_solve

                lhs, rhs = None, sympify(0)
                if '=' in equation_str_solve: # Treat single '=' as '==' for equation solving
                    lhs_str, rhs_str = equation_str_solve.split('=', 1)
                    lhs = parse_expression_safely(lhs_str.strip(), local_dict=current_parse_local_dict)
                    rhs = parse_expression_safely(rhs_str.strip(), local_dict=current_parse_local_dict)
                                
                eq_display = Equality(lhs, rhs)
                expr_to_solve = lhs - rhs

                st.write("Equation to Solve:")
                st.latex(latex(eq_display))

                solutions = solve(expr_to_solve, var_solve)

                st.write(f"Solutions for ${latex(var_solve)}$:")
                if solutions:
                    st.latex(latex(solutions))
                else:
                    st.warning("No symbolic solutions found. The equation might have no solution, solutions involve complex numbers not displayed.")

            except (SympifyError, SyntaxError) as e:
                st.error(f"Error parsing equation: {e}. Please check the syntax. Ensure variable is one of {list(self.common_symbols.keys())} or uniquely identifiable.")
            except Exception as e:
                st.error(f"An unexpected error occurred during solving: {e}")

# --- Equation Plotter (y = f(x)) ---
class EquationPlotter(MathOperation):
    def display_ui_and_process(self):
        st.header("📊 Graphing Calculator ( Simple )")
        equation_str_plot = st.text_input(
            "Enter an expression in terms of 'x' for y (e.g., sin(x)/x, tan(x), log(x)):",
            value="sin(x)/x",
            key="plot_eq"
        )
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("Minimum x value:", value=-10.0, step=0.5, format="%.2f", key="plot_xmin")
        with col2:
            x_max = st.number_input("Maximum x value:", value=10.0, step=0.5, format="%.2f", key="plot_xmax")
        num_points = st.slider("Number of points for plotting:", min_value=50, max_value=2000, value=400, step=50, key="plot_num_points")

        if st.button("Plot Graph", key="plot_button"):
            if not equation_str_plot:
                st.warning("Please enter an expression to plot.")
                return
            if x_min >= x_max:
                st.warning("Minimum x value must be less than maximum x value.")
                return

            try:
                current_parse_local_dict = self.parse_local_dict.copy()
                # Ensure 'x' is treated as the symbol x for this plot
                current_parse_local_dict['x'] = self.x 
                expr_plot = parse_expression_safely(equation_str_plot, local_dict=current_parse_local_dict)
                
                st.write("Equation to Plot: $y = " + latex(expr_plot) + "$")

                func = lambdify(self.x, expr_plot, modules=['numpy', {'log': np.log, 'sqrt': np.sqrt, 'abs':np.abs}]) 

                x_vals = np.linspace(float(x_min), float(x_max), int(num_points))
                y_vals = np.full_like(x_vals, np.nan)

                try:
                    calculated_y = func(x_vals)
                    if np.iscomplexobj(calculated_y):
                        st.info("Function resulted in complex numbers. Plotting the real part.")
                        calculated_y = np.real(calculated_y)
                    y_vals = np.where(np.isfinite(calculated_y), calculated_y, np.nan)
                except (TypeError, ValueError, NameError, AttributeError, ZeroDivisionError) as eval_err:
                    st.error(f"Error evaluating the function for plotting: {eval_err}. Check syntax (e.g., log(x) for x<=0).")
                    y_vals = np.full_like(x_vals, np.nan) # Ensure no plot on error


                if np.all(np.isnan(y_vals)):
                    st.warning("Function resulted in invalid values (NaN or Inf) for the entire plotting range.")
                    return

                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals, label=f'$y = {latex(expr_plot)}$')
                self._plot_(ax, f"Plot of $y = {latex(expr_plot)}$", y_vals_list=[y_vals])
                st.pyplot(fig)

            except (SympifyError, SyntaxError) as e:
                st.error(f"Error parsing expression: {e}. Please check the syntax. Ensure you are using 'x' as the variable.")
            except Exception as e:
                st.error(f"An unexpected error occurred during plotting setup: {e}")

# --- Inequality Plotter ---
class InequalityPlotter(MathOperation):
    def display_ui_and_process(self):
        st.header("📉 Inequality Plotter")
        inequality_str = st.text_input(
            "Enter an inequality with x and y (e.g., y > x**2, x + y <= 0, x**2/4 + y**2/9 < 1):",
            value="y > x**2",
            key="ineq_str"
        )

        st.write("Define plot range:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("Min x:", value=-5.0, step=0.5, format="%.2f", key="ineq_xmin")
        with col2:
            x_max = st.number_input("Max x:", value=5.0, step=0.5, format="%.2f", key="ineq_xmax")
        with col3:
            y_min = st.number_input("Min y:", value=-5.0, step=0.5, format="%.2f", key="ineq_ymin")
        with col4:
            y_max = st.number_input("Max y:", value=5.0, step=0.5, format="%.2f", key="ineq_ymax")
        
        num_of_points = st.slider(
            "Grid density for plotting inequality:", 
            min_value=50, max_value=500, value=100, step=10, 
            key="ineq_numpoints",
            help="Higher density means smoother plot but slower computation."
        )

        if st.button("Plot Inequality", key="plot_ineq_button"):
            if not inequality_str:
                st.warning("Please enter an inequality to plot.")
                return
            if x_min >= x_max or y_min >= y_max:
                st.warning("Min values must be less than Max values for the plot range.")
                return

            match = re.match(r"(.+?)\s*(>=|<=|>|<)\s*(.+)", inequality_str)
            if not match:
                st.error("Invalid inequality format. Use standard operators like >, <, >=, <= (e.g., y > x**2).")
                return

            lhs_str, operator, rhs_str = match.groups()

            try:
                current_parse_local_dict = self.parse_local_dict.copy()
                lhs_expr = parse_expression_safely(lhs_str, local_dict=current_parse_local_dict)
                rhs_expr = parse_expression_safely(rhs_str, local_dict=current_parse_local_dict)

                eval_expr = lhs_expr - rhs_expr
                st.write(f"Plotting region where: ${latex(lhs_expr)} {operator} {latex(rhs_expr)}$")
                st.write(f"(Equivalent to: ${latex(eval_expr)} {'< 0' if operator in ['<', '<='] else '> 0'}$ etc.)")

                numpy_modules = ['numpy', {
                    'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
                    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                    'exp': np.exp
                }]
                eval_func = lambdify((self.x, self.y), eval_expr, modules=numpy_modules)

                x_grid_vals = np.linspace(float(x_min), float(x_max), int(num_of_points))
                y_grid_vals = np.linspace(float(y_min), float(y_max), int(num_of_points))
                X_grid, Y_grid = np.meshgrid(x_grid_vals, y_grid_vals)

                Z_vals = np.full_like(X_grid, np.nan, dtype=float)
                try:
                    Z_vals = eval_func(X_grid, Y_grid)
                except Exception as e: 
                    st.error(f"Error during grid evaluation: {e}. This can happen with functions like log(negative).")
                    return
                
                fig, ax = plt.subplots()

                if operator == '>':
                    condition = Z_vals > 1e-9 # Add tolerance for strict inequalities
                elif operator == '<':
                    condition = Z_vals < -1e-9 # Add tolerance
                elif operator == '>=':
                    condition = Z_vals >= -1e-9 # Tolerance for floating point comparisons near zero
                elif operator == '<=':
                    condition = Z_vals <= 1e-9  # Tolerance
                
                ax.contourf(X_grid, Y_grid, condition.astype(float), levels=[0.5, 1.5], colors=['#1f77b460'], extend='neither')
                ax.contour(X_grid, Y_grid, Z_vals, levels=[0], colors='k', linewidths=1.5)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                self._plot_(ax, f"Inequality: ${latex(lhs_expr)} {operator} {latex(rhs_expr)}$", y_vals_list=None, x_label="$x$", y_label="$y$")
                if ax.get_legend() is not None: # Remove default legend if _plot_beautify added one
                    ax.legend().remove()
                st.pyplot(fig)

            except (SympifyError, SyntaxError) as e:
                st.error(f"Error parsing inequality: {e}. Ensure 'x' and 'y' are used for variables.")
            except TypeError as te:
                 st.error(f"Error in expression (often due to undefined functions or variables): {te}. Ensure all functions are standard or defined.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- System of Equations Plotter ---
class SystemOfEquationsPlotter(MathOperation):
    def display_ui_and_process(self):
        st.header("📈 System of Equations Solver")
        st.write("Enter two equations in the form $y = \\text{expression in } x$.")

        eq1_str = st.text_input(
            "Equation 1 (e.g., x**2 or sin(x)):",
            value="x**2 - 2",
            key="sys_eq1_y_expr"
        )
        eq2_str = st.text_input(
            "Equation 2 (e.g., -x + 1 or cos(x)):",
            value="-x + 0",
            key="sys_eq2_y_expr"
        )

        col1, col2 = st.columns(2)
        with col1:
            x_min_system = st.number_input("Minimum x value:", value=-5.0, step=0.5, format="%.2f", key="sys_xmin")
        with col2:
            x_max_sys = st.number_input("Maximum x value:", value=5.0, step=0.5, format="%.2f", key="sys_xmax")
        
        num_of_points_system = st.slider(
            "Number of points for plotting system:", 
            min_value=50, max_value=1000, value=200, step=50, 
            key="sys_numpoints"
        )

        if st.button("Plot System and Find Intersections", key="plot_sys_button"):
            if not eq1_str or not eq2_str:
                st.warning("Please enter both expressions for y.")
                return
            if x_min_system >= x_max_sys:
                st.warning("Minimum x value must be less than maximum x value.")
                return

            try:
                current_parse_local_dict = self.parse_local_dict.copy()
                current_parse_local_dict['x'] = self.x 

                y1_expr = parse_expression_safely(eq1_str, local_dict=current_parse_local_dict)
                y2_expr = parse_expression_safely(eq2_str, local_dict=current_parse_local_dict)

                st.write("Equations to Plot:")
                st.latex(f"y_1 = {latex(y1_expr)}")
                st.latex(f"y_2 = {latex(y2_expr)}")

                numpy_modules = ['numpy', {'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs}]
                y1_func = lambdify(self.x, y1_expr, modules=numpy_modules)
                y2_func = lambdify(self.x, y2_expr, modules=numpy_modules)

                x_vals = np.linspace(float(x_min_system), float(x_max_sys), int(num_of_points_system))
                
                y1_vals = np.full_like(x_vals, np.nan)
                y2_vals = np.full_like(x_vals, np.nan)

                try:
                    calc_y1 = y1_func(x_vals)
                    if np.iscomplexobj(calc_y1): calc_y1 = np.real(calc_y1)
                    y1_vals = np.where(np.isfinite(calc_y1), calc_y1, np.nan)
                except Exception: pass 

                try:
                    calc_y2 = y2_func(x_vals)
                    if np.iscomplexobj(calc_y2): calc_y2 = np.real(calc_y2)
                    y2_vals = np.where(np.isfinite(calc_y2), calc_y2, np.nan)
                except Exception: pass

                if np.all(np.isnan(y1_vals)) and np.all(np.isnan(y2_vals)):
                    st.warning("Both functions resulted in invalid values for the entire plotting range.")
                    return
                
                fig, ax = plt.subplots()
                if not np.all(np.isnan(y1_vals)):
                    ax.plot(x_vals, y1_vals, label=f'$y_1 = {latex(y1_expr)}$')
                if not np.all(np.isnan(y2_vals)):
                    ax.plot(x_vals, y2_vals, label=f'$y_2 = {latex(y2_expr)}$', linestyle='--')
                
                st.write("---")
                st.markdown("Intersections (Symbolic):")
                intersects = []
                try:
                    intersect_eq = Eq(y1_expr, y2_expr)
                    x_solutions = solve(intersect_eq, self.x) 
                    
                    if not x_solutions:
                        st.info("No symbolic real intersection points found by SymPy's `solve`.")
                    else:
                        st.write(f"Solutions for $x$ where ${latex(y1_expr)} = {latex(y2_expr)}$:")
                        st.latex(latex(x_solutions))
                        
                        for x_sol_expr in x_solutions:
                            try:
                                x_sol_val = N(x_sol_expr) 
                                if 'I' in str(x_sol_val) or (hasattr(x_sol_expr, 'is_real') and x_sol_expr.is_real is False):
                                     st.write(f"Skipping complex solution $x = {latex(x_sol_expr)}$")
                                     continue

                                if float(x_min_system) <= float(x_sol_val) <= float(x_max_sys):
                                    y_sol_val_expr = y1_expr.subs(self.x, x_sol_expr)
                                    y_sol_val = N(y_sol_val_expr)

                                    intersects.append((float(x_sol_val), float(y_sol_val)))
                                    st.write(f"Found intersection at: $(x, y) \\approx ({float(x_sol_val):.4f}, {float(y_sol_val):.4f})$")
                                else:
                                    st.write(f"Solution $x \\approx {float(x_sol_val):.4f}$ is outside the current plot's x-range.")
                            except (TypeError, AttributeError, ValueError, RuntimeError) as e: # Added RuntimeError
                                st.warning(f"Could not numerically evaluate or substitute solution ${latex(x_sol_expr)}$: {e}")

                        if intersects:
                            xs, ys = zip(*intersects)
                            ax.plot(xs, ys, 'ro', label='Intersections') 
                        
                except NotImplementedError:
                    st.warning("SymPy's `solve` could not handle this system of equations.")
                except Exception as e:
                    st.error(f"Error finding intersections: {e}")

                self._plot_(ax, f"System: ${latex(Eq(Symbol('y'), y1_expr))}$ & ${latex(Eq(Symbol('y'), y2_expr))}$", y_vals_list=[y1_vals, y2_vals])
                st.pyplot(fig)

            except (SympifyError, SyntaxError) as e:
                st.error(f"Error parsing one of the expressions: {e}. Please use 'x' as the variable.")
            except Exception as e:
                st.error(f"An unexpected error occurred during system plotting: {e}")


# --- Main App Structure ---
class AdvancedMathApp:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Advanced Math Tool")
        st.title("🧮 Advanced Math Tool")
        st.write("Perform differentiation, solve equations, plot graphs, inequalities, and systems of equations.")
        
        self.operations = {
            "Calculate Derivative": DerivativeCalculator(),
            "Solve Single Variable Equation": EquationSolver(),
            "Graph an Equation ": EquationPlotter(),
            "Plot Inequality (x, y)": InequalityPlotter(),
            "Solve System of Equations": SystemOfEquationsPlotter(),
        }

    def run(self):
        st.sidebar.title("Select Operation")
        
        operation_names = list(self.operations.keys())
        chosen_operation_name = st.sidebar.radio(
            "Choose what you want to do:",
            operation_names,
            key="main_operation_selector"
        )
        operation_handler = self.operations[chosen_operation_name]
        operation_handler.display_ui_and_process()
# --- Run the app ---
if __name__ == "__main__":
    app = AdvancedMathApp()
    app.run()