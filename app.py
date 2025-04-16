# Import necessary libraries for mathematical operations, plotting, and Streamlit app
import cmath
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.linalg import sqrtm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.graph_objects as go


# Set the configuration for the Streamlit page
st.set_page_config(page_title="Plot Your Equation", layout="wide")

# Set the title of the Streamlit app
st.title("Explore Your Functions Interactively")

# Sidebar for user controls
st.sidebar.header("Parameters")
# Dropdown menu for selecting the type of equation to explore
equation_type = st.sidebar.selectbox(
    "Select Your Eqaution",
    ["Wigner Function (n) Fock States", "Gaussian Wigner Function 2D",
     "Coupling Matrix", "Gaussian Wigner Function 4D", "Wasserstein Distance"]
)

# Define the Wigner function for a quantum harmonic oscillator


def wigner_function(n, x, p):
    """
    EN: Calculates the Wigner function for a quantum harmonic oscillator using Laguerre polynomials. 
    This function takes the quantum number n and two phase space variables (position x and momentum p), 
    and returns the Wigner distribution value at each (x, p) point.

    FR : Calcule la fonction de Wigner pour un oscillateur harmonique quantique en utilisant les polynômes de Laguerre. 
    Cette fonction prend en entrée le nombre quantique n ainsi que deux variables d’espace de phase (position x et quantité de mouvement p), 
    et retourne la valeur de la distribution de Wigner pour chaque point (x, p).
    """
    # Calculate the prefactor of the Wigner function
    prefactor = (-1)**n / np.pi
    # Calculate the exponential term
    exponent = np.exp(-x**2 - p**2)
    # Calculate the Laguerre polynomial term
    laguerre_term = sp.eval_genlaguerre(n, 0, 2 * (x**2 + p**2))
    # Return the Wigner function value
    return prefactor * exponent * laguerre_term

# Define the 4D Gaussian Wigner function for a bipartite quantum system


def wigner_gaussian4D(x1, p1, x2, p2, gamma):
    """
    EN: Computes the 4D Gaussian Wigner function for a bipartite (two-mode) quantum system characterized by a 4x4 covariance matrix gamma. 
    The function evaluates the Wigner function at a specific point (x1, p1, x2, p2) in the four-dimensional phase space, 
    assuming the system satisfies the quantum uncertainty principle.

    FR : Calcule la fonction de Wigner gaussienne 4D pour un système quantique biparti (à deux modes) à partir d’une matrice de covariance 4x4 gamma. 
    La fonction évalue la fonction de Wigner à un point spécifique (x1, p1, x2, p2) de l’espace de phase à quatre dimensions, 
    en supposant que le système satisfait au principe d’incertitude quantique.
    """
    # Check if the covariance matrix satisfies the uncertainty principle
    if uncertainty4D(gamma):
        # Create vector r from the input coordinates
        vec_r = np.array([[x1, p1, x2, p2]]).reshape(-1, 1)

        # Transpose of vec_r
        vec_r_t = np.transpose(vec_r)

        # Calculate the determinant and inverse of the covariance matrix gamma
        det = np.linalg.det(gamma)
        inv = np.linalg.inv(gamma)

        # Calculate the normalization factor
        norm = 1 / ((2 * np.pi)**2 * np.sqrt(det))
        # Calculate the exponent term
        exponent = -0.5 * vec_r_t @ inv @ vec_r

        # Debugging prints for vector r and its transpose
        # print(f"vec_r T: {vec_r_t.flatten()}")
        # print("\n")
        # print(f"vec_r: {vec_r}")
        # print("\n")

        # Return the Wigner function value
        return norm * np.exp(exponent)
    else:
        # Return a message if the covariance matrix does not satisfy the uncertainty principle
        return "Gamma doesn't respect uncertainty relation!"

# Define the 2D Gaussian Wigner function


def wigner_gaussian(r, covar_mx):
    """
    EN: Computes the value of the 2D Gaussian Wigner function at a given point r = (x, p), 
    using a 2x2 covariance matrix that describes the quantum state's spread and correlation in phase space. 
    This is used to model single-mode Gaussian states.

    FR : Calcule la valeur de la fonction de Wigner gaussienne 2D en un point donné r = (x, p), 
    en utilisant une matrice de covariance 2x2 décrivant l’étendue et les corrélations de l’état quantique dans l’espace de phase. 
    Cette fonction est utilisée pour modéliser des états gaussiens à un seul mode.
    """
    # Calculate the determinant and inverse of the covariance matrix
    det = np.linalg.det(covar_mx)
    inv = np.linalg.inv(covar_mx)
    # Calculate the normalization factor
    norm = 1 / (2 * np.pi * np.sqrt(det))
    # Calculate the exponent term
    exponent = -0.5 * r @ inv @ r
    # Return the Wigner function value
    return norm * np.exp(exponent)

# Check if a 4x4 covariance matrix satisfies the uncertainty principle


def uncertainty4D(gamma):
    """
    EN: Checks whether the input covariance matrix gamma respects the generalized uncertainty principle 
    by computing the matrix M = gamma + (i/2) * Omega and verifying if it is Hermitian and positive semi-definite. 
    This ensures that the quantum state represented by gamma is physically valid.

    FR : Vérifie si la matrice de covariance gamma respecte le principe d’incertitude généralisé 
    en calculant la matrice M = gamma + (i/2) * Omega et en vérifiant si elle est hermitienne et semi-définie positive. 
    Cela garantit que l’état quantique représenté par gamma est physiquement valide.
    """
    # Convert gamma to a complex numpy array
    gamma = np.array(gamma, dtype=complex)
    dim = gamma.shape[0]

    # Check if gamma is a square matrix and its dimension is even
    if gamma.shape[1] != dim or (dim % 2 != 0):
        return False

    n = dim // 2

    # Define the symplectic form Omega
    Omega = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0]
    ], dtype=complex)

    # Form the matrix M = gamma + (i/2)*Omega
    M = gamma + 1j/2 * Omega

    # Check if M is Hermitian
    if not np.allclose(M, M.conj().T):
        return False

    # Check if M is positive semidefinite
    eigenvals = np.linalg.eigvalsh(M)  # eigvalsh is for Hermitian matrices
    # Allow a tiny numerical tolerance
    return np.all(eigenvals > -1e-14)

# Check if a 2x2 covariance matrix satisfies the uncertainty principle


def uncertainty2D(matrix_covariance):
    # Define the imaginary unit
    i = cmath.sqrt(-1)
    # Define the symplectic form for 2D
    w = np.array([[0, 1], [-1, 0]], dtype=float)

    # Form the matrix M = matrix_covariance + (i/2)*w
    M = np.array(matrix_covariance, dtype=complex) + (i * w / 2)

    # Compute eigenvalues and check if all are >= 0 (positive semidefinite)
    eigenvals = np.linalg.eigvals(M)
    return np.all(eigenvals >= 0)

# Calculate the minimal coupling matrix for two 2x2 covariance matrices


def minimal_coupling(A, B):
    """
    Given two 2x2 covariance matrices A and B (which must be valid
    quantum covariance matrices), returns the minimal coupling X 
    such that the full 4x4 block matrix

        gamma = [[ A,  X ],
                 [ X,  B ]]

    is a valid covariance matrix for the joint (two-mode) Gaussian state.

    We use X = sqrt( sqrt(A) * B * sqrt(A) ).
    """
    # Verify that matrices A & B respect the uncertainty principle
    if uncertainty2D(A) and uncertainty2D(B):
        # Compute the positive square root of A
        sqrtA = sqrtm(A)
        # Compute X = sqrt( sqrt(A) * B * sqrt(A) )
        X = sqrtm(sqrtA @ B @ sqrtA)
        # Clean up any tiny imaginary parts due to numerical precision
        X = np.real_if_close(X)
        return X
    else:
        # Display a message if matrices do not respect the uncertainty principle
        st.write("Your Matrix Doesn't Respect the Uncertainty Relation 1")

# Calculate the Wasserstein distance between two covariance matrices


def wasserstein_distance(matrix_A, matrix_B):
    # Find the optimal coupling matrix X
    matrix_X = minimal_coupling(matrix_A, matrix_B)

    # Compute the traces of matrices A, B, and X
    trace_a = np.trace(matrix_A)
    trace_b = np.trace(matrix_B)
    trace_x = np.trace(matrix_X)

    # Compute the Wasserstein distance
    wd = (0.5 * trace_a) + (0.5 * trace_b) - trace_x
    return wd


# Add a separator and subheader in the sidebar for equation information
st.sidebar.markdown("---")
st.sidebar.subheader("Equation Info")

# Display information about the selected equation type in the sidebar
if equation_type == "Wigner Function (n) Fock States":
    st.sidebar.latex(
        r"W_n(x,p) = \frac{(-1)^n}{\pi} e^{-(x^2+p^2)} L_n(2(x^2+p^2))")
    st.sidebar.write("where Lₙ is the Laguerre polynomial of degree n")

elif equation_type == "Gaussian Wigner Function 2D":
    st.sidebar.latex(
        r"W(r) = \frac{1}{2\pi\sqrt{\det(\gamma)}} e^{-\frac{1}{2}r^T \gamma^{-1} r}")
    st.sidebar.write("where r = (x,p) and γ is the covariance matrix")

elif equation_type == "Gaussian Wigner Function 4D":
    st.sidebar.latex(
        r"W(r) = \frac{1}{2\pi\sqrt{\det(\gamma)}} e^{-\frac{1}{2}r^T \gamma^{-1} r}")
    st.sidebar.write("where r = (x,p) and γ is the covariance matrix")

elif equation_type == "Coupling Matrix":
    st.sidebar.write(
        "The coupling matrix represents the correlation between two Gaussian quantum states in phase space.")

# Add a section for custom equation input (coming soon)
st.sidebar.markdown("---")
st.sidebar.subheader("Custom Equation")
st.sidebar.write("Coming soon: Add your own equation")

# Add instructions for using the app
st.sidebar.markdown("---")
with st.sidebar.expander("Instructions"):
    st.write("""
    1. Select an equation type from the dropdown
    2. Adjust parameters using the sliders
    3. View the resulting plot in real-time
    4. Change plot type or colormap as needed
    5. For Wigner Function, you can download the data as CSV
    """)

# Main content area for displaying plots and results
if equation_type == "Wigner Function (n) Fock States":
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    # Contents of the first column
    with col1:
        # Slider for selecting the quantum number n
        n = st.slider("Quantum Number (n)", 0, 20, 5)

        # Slider for selecting the resolution of the plot
        resolution = st.slider("Resolution", 20, 200, 100)

        # Slider for selecting the range of the x-axis
        x_axes_range = st.slider("X Range", -5.0, 5.0, (-3.0, 3.0))

        # Slider for selecting the range of the p-axis
        p_axes_range = st.slider("P Range", -5.0, 5.0, (-3.0, 3.0))

        # Radio buttons for selecting the plot type
        plot_type = st.radio("Plot Type", ["Surface", "Contour"])
        # Dropdown for selecting the colormap
        colormap = st.selectbox(
            "Color Map", ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])

    # Contents of the second column
    with col2:
        # Define the grid for the plot using the selected resolution and range
        x_values = np.linspace(x_axes_range[0], x_axes_range[1], resolution)
        p_values = np.linspace(p_axes_range[0], p_axes_range[1], resolution)
        x_grid, p_grid = np.meshgrid(x_values, p_values)

        # Compute the Wigner function values on the grid
        wigner_vals = wigner_function(n, x_grid, p_grid)

        # Plot the Wigner function as a surface or contour plot
        if plot_type == "Surface":
            fig = go.Figure(
                data=[go.Surface(x=x_grid, y=p_grid, z=wigner_vals, colorscale=colormap)])
            fig.update_layout(
                title=f"Wigner Function for n = {n}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="p",
                    zaxis_title="$W_n(x, p)$"
                )
            )
            st.plotly_chart(fig)
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            contour = ax.contourf(
                x_grid, p_grid, wigner_vals, cmap=colormap, levels=50)
            ax.set_xlabel("x")
            ax.set_ylabel("p")
            ax.set_title(f"Wigner Function for n = {n}")
            st.pyplot(fig)

        # Button for generating and downloading the data as a CSV file
        if st.button("Generate Download Data"):
            df = pd.DataFrame({
                'x': x_grid.flatten(),
                'p': p_grid.flatten(),
                'wigner_value': wigner_vals.flatten()
            })
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"wigner_function_n{n}.csv",
                mime="text/csv"
            )

elif equation_type == "Gaussian Wigner Function 2D":
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    # Contents of the first column
    with col1:
        st.subheader("Covariance Matrix")
        # Sliders for setting the covariance matrix elements
        covar_xx = st.slider("Var(x)", 0.1, 3.0, 1.0, 0.1)
        covar_pp = st.slider("Var(p)", 0.1, 3.0, 1.0, 0.1)
        covar_xp = st.slider("Cov(x,p)", -1.0, 1.0, 0.0, 0.1)

        # Slider for selecting the resolution of the plot
        resolution = st.slider("Resolution", 20, 200, 100)
        # Sliders for selecting the range of the x and p axes
        x_range = st.slider("X Range", -5.0, 5.0, (-3.0, 3.0))
        p_range = st.slider("P Range", -5.0, 5.0, (-3.0, 3.0))

        # Radio buttons for selecting the plot type
        plot_type = st.radio("Plot Type", ["Surface", "Contour"])
        # Dropdown for selecting the colormap
        colormap = st.selectbox(
            "Color Map", ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])

    # Contents of the second column
    with col2:
        # Construct the covariance matrix from the slider values
        covar_mx = np.array([[covar_xx, covar_xp], [covar_xp, covar_pp]])

        # Check if the covariance matrix is positive-definite
        if np.linalg.det(covar_mx) <= 0:
            st.warning(
                "Warning: Current values do not form a valid covariance matrix. Please adjust parameters.")

        # Display the covariance matrix
        st.write("Covariance Matrix:")
        st.write(covar_mx)

        # Set up a grid over phase space (x and p) using the selected resolution and range
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        p_vals = np.linspace(p_range[0], p_range[1], resolution)
        X, P = np.meshgrid(x_vals, p_vals)

        # Evaluate the Wigner function on the grid
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                r = np.array([X[i, j], P[i, j]])
                Z[i, j] = wigner_gaussian(r, covar_mx)

        # Plot the Wigner function as a surface or contour plot
        if plot_type == "Surface":
            fig = go.Figure(
                data=[go.Surface(x=X, y=P, z=Z, colorscale=colormap)])
            fig.update_layout(
                title="Gaussian Wigner Function",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="p",
                    zaxis_title="Wigner Function"
                )
            )
            st.plotly_chart(fig)
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            contour = ax.contourf(X, P, Z, cmap=colormap, levels=50)
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel("x")
            ax.set_ylabel("p")
            ax.set_title("Gaussian Wigner Function")
            st.pyplot(fig)

elif equation_type == "Coupling Matrix":
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    # Contents of the first column
    with col1:
        st.subheader("State 1 Covariance Matrix")
        # Sliders for setting the covariance matrix elements of state 1
        gamma1_xx = st.slider("γ1 Var(x)", 0.1, 3.0, 1.2, 0.1)
        gamma1_pp = st.slider("γ1 Var(p)", 0.1, 3.0, 1.1, 0.1)
        gamma1_xp = st.slider("γ1 Cov(x,p)", -1.0, 1.0, 0.5, 0.1)

        st.subheader("State 2 Covariance Matrix")
        # Sliders for setting the covariance matrix elements of state 2
        gamma2_xx = st.slider("γ2 Var(x)", 0.1, 3.0, 1.0, 0.1)
        gamma2_pp = st.slider("γ2 Var(p)", 0.1, 3.0, 1.3, 0.1)
        gamma2_xp = st.slider("γ2 Cov(x,p)", -1.0, 1.0, 0.3, 0.1)

        st.subheader("Coupling Matrix")
        # Sliders for setting the coupling matrix elements
        X_xx = st.slider("X(x1,x2)", -1.0, 1.0, 0.6, 0.1)
        X_pp = st.slider("X(p1,p2)", -1.0, 1.0, 0.7, 0.1)
        X_xp = st.slider("X(x1,p2)", -1.0, 1.0, 0.2, 0.1)
        X_px = st.slider("X(p1,x2)", -1.0, 1.0, 0.2, 0.1)

    # Contents of the second column
    with col2:
        # Define covariance matrices for two Gaussian states
        gamma_1 = np.array([[gamma1_xx, gamma1_xp], [gamma1_xp, gamma1_pp]])
        gamma_2 = np.array([[gamma2_xx, gamma2_xp], [gamma2_xp, gamma2_pp]])

        # Define the optimal coupling matrix
        X_optimal = np.array([[X_xx, X_xp], [X_px, X_pp]])

        # Construct the full coupling matrix γ
        gamma = np.block([[gamma_1, X_optimal], [X_optimal.T, gamma_2]])

        # Verify that the matrix respects the uncertainty relation
        if uncertainty4D(gamma) == True:
            # Create a meshgrid for plotting
            X, Y = np.meshgrid(range(gamma.shape[0]), range(gamma.shape[1]))
            # Create a 3D surface plot of the coupling matrix
            fig = go.Figure(
                data=[go.Surface(x=X, y=Y, z=gamma, colorscale='plasma')])
            fig.update_layout(
                title="Interactive 3D Heatmap of the Coupling Matrix for Gaussian States",
                scene=dict(
                    xaxis=dict(title="Basis State 1", tickmode='array', tickvals=list(
                        range(4)), ticktext=["Q1", "P1", "Q2", "P2"]),
                    yaxis=dict(title="Basis State 2", tickmode='array', tickvals=list(
                        range(4)), ticktext=["Q1", "P1", "Q2", "P2"]),
                    zaxis=dict(title="Coupling Strength")
                )
            )
            st.plotly_chart(fig)
        else:
            # Display a message if the matrix does not respect the uncertainty relation
            st.write("Your Matrix Doesn't Respect the Uncertainty Relation 1")

        # Display the numerical matrix
        st.write("Full Coupling Matrix:")
        st.write(pd.DataFrame(
            gamma,
            columns=["Q1", "P1", "Q2", "P2"],
            index=["Q1", "P1", "Q2", "P2"]
        ))

elif equation_type == "Gaussian Wigner Function 4D":
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    # Contents of the first column
    with col1:
        st.subheader("Vector r (x1, p1, x2, p2)")
        # Sliders for setting the components of vector r
        vecr_x1 = st.slider("x1", -5.0, 5.0, 0.0, 0.1)
        vecr_p1 = st.slider("p1", -5.0, 5.0, 0.0, 0.1)
        vecr_x2 = st.slider("x2", -5.0, 5.0, 0.0, 0.1)
        vecr_p2 = st.slider("p2", -5.0, 5.0, 0.0, 0.1)

        st.subheader("Matrix A (Top-Left, State A)")
        # Sliders for setting the covariance matrix elements of matrix A
        A_xx = st.slider("A[0,0] (Var x1)", 0.1, 3.0, 1.0, 0.1)
        A_pp = st.slider("A[1,1] (Var p1)", 0.1, 3.0, 1.0, 0.1)
        A_xp = st.slider("A[0,1] and A[1,0] (Cov x1,p1)", -1.0, 1.0, 0.0, 0.1)

        st.subheader("Matrix B (Bottom-Right, State B)")
        # Sliders for setting the covariance matrix elements of matrix B
        B_xx = st.slider("B[0,0] (Var x2)", 0.1, 3.0, 1.0, 0.1)
        B_pp = st.slider("B[1,1] (Var p2)", 0.1, 3.0, 1.0, 0.1)
        B_xp = st.slider("B[0,1] and B[1,0] (Cov x2,p2)", -1.0, 1.0, 0.0, 0.1)

        st.subheader("Covariances (Bottom-Left, Top-Right)")
        # Sliders for setting the covariance elements between matrices A and B
        co1 = st.slider("co1", -5.0, 5.0, 0.0, 0.1)
        co2 = st.slider("co2", -5.0, 5.0, 0.0, 0.1)
        co3 = st.slider("co3", -5.0, 5.0, 0.0, 0.1)
        co4 = st.slider("co4", -5.0, 5.0, 0.0, 0.1)

        # Radio buttons for selecting the view mode
        view_mode = st.radio("View Mode", ["2D Slice", "3D Plot", "4D Sweep"])
        st.subheader("Slice Selection")
        # Dropdown for selecting the 2D slice to plot
        slice_axes = st.selectbox("Select 2D slice to plot", [
            ("x1", "p1"),
            ("x1", "x2"),
            ("x1", "p2"),
            ("p1", "x2"),
            ("p1", "p2"),
            ("x2", "p2")
        ])

        # Dropdown for selecting the colormap
        colormap = st.selectbox(
            "Color Map", ["viridis", "plasma", "inferno", "magma", "cividis"])

    # Contents of the second column
    with col2:
        # Build vector r from the slider values
        vector_r = np.array([[vecr_x1], [vecr_p1], [vecr_x2], [vecr_p2]])

        # Build matrices A & B from the slider values
        matrix_A = np.array([[A_xx, A_xp], [A_xp, A_pp]])
        matrix_B = np.array([[B_xx, B_xp], [B_xp, B_pp]])

        # Build the covariance matrix using minimal coupling
        covar_matrix = np.block([[co1, co2], [co3, co4]])
        gamma = np.block([[matrix_A, covar_matrix],
                          [covar_matrix, matrix_B]])

        # Display the current vector r
        st.subheader("Current Vector r")
        st.dataframe(pd.DataFrame(vector_r, index=[
            "x1", "p1", "x2", "p2"], columns=["Value"]))

        # Calculate the Wigner function value at the current vector r
        result = wigner_gaussian4D(
            vecr_x1, vecr_p1, vecr_x2, vecr_p2, gamma)
        st.subheader("Wigner Function Value at r")
        st.write(result)

        st.subheader("Matrix A Visualization")
        # Display matrix A as a dataframe
        st.dataframe(pd.DataFrame(matrix_A, columns=[
            "x", "p"], index=["x", "p"]))

        st.subheader("Matrix B Visualization")
        # Display matrix B as a dataframe
        st.dataframe(pd.DataFrame(matrix_B, columns=[
            "x", "p"], index=["x", "p"]))

        # Display the current covariance matrix γ
        st.subheader("Current Covariance Matrix γ")
        st.dataframe(pd.DataFrame(gamma, columns=[
                     "x1", "p1", "x2", "p2"], index=["x1", "p1", "x2", "p2"]))

        # Plot the Wigner function based on the selected view mode
        if view_mode == "2D Slice":
            st.subheader(
                f"Wigner Function Slice: {slice_axes[0]} vs {slice_axes[1]}")
            resolution = 100
            axis_vals = np.linspace(-3, 3, resolution)
            X, Y = np.meshgrid(axis_vals, axis_vals)
            Z = np.zeros_like(X)

            # Evaluate the Wigner function on the grid for the selected slice
            for i in range(resolution):
                for j in range(resolution):
                    coords = {"x1": vecr_x1, "p1": vecr_p1,
                              "x2": vecr_x2, "p2": vecr_p2}
                    coords[slice_axes[0]] = X[i, j]
                    coords[slice_axes[1]] = Y[i, j]
                    Z[i, j] = wigner_gaussian4D(
                        coords["x1"], coords["p1"], coords["x2"], coords["p2"], gamma
                    )

            # Create a contour plot for the selected slice
            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(X, Y, Z, cmap=colormap, levels=50)
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel(slice_axes[0])
            ax.set_ylabel(slice_axes[1])
            st.pyplot(fig)

        elif view_mode == "3D Plot":
            st.subheader(
                f"Interactive Wigner Function 3D Surface: {slice_axes[0]} vs {slice_axes[1]}")
            resolution = 100
            axis_vals = np.linspace(-3, 3, resolution)
            X, Y = np.meshgrid(axis_vals, axis_vals)
            Z = np.zeros_like(X)

            # Evaluate the Wigner function on the grid for the 3D plot
            for i in range(resolution):
                for j in range(resolution):
                    coords = {"x1": vecr_x1, "p1": vecr_p1,
                              "x2": vecr_x2, "p2": vecr_p2}
                    coords[slice_axes[0]] = X[i, j]
                    coords[slice_axes[1]] = Y[i, j]
                    Z[i, j] = wigner_gaussian4D(
                        coords["x1"], coords["p1"], coords["x2"], coords["p2"], gamma
                    )

            # Create a 3D surface plot for the selected slice
            fig = go.Figure(
                data=[go.Surface(x=X, y=Y, z=Z, colorscale=colormap)])
            fig.update_layout(
                title=f"Interactive Wigner Function 3D Surface: {slice_axes[0]} vs {slice_axes[1]}",
                scene=dict(
                    xaxis_title=slice_axes[0],
                    yaxis_title=slice_axes[1],
                    zaxis_title="Wigner Value"
                )
            )
            st.plotly_chart(fig)

        st.subheader("Wasserstein Distance Visualization")
        # Compute the Wasserstein distance between matrix A and matrix B
        distance = wasserstein_distance(matrix_A, matrix_B)
        max_range = distance * 1.2 if distance > 0 else 1.0
        # Create an interactive gauge indicator for the Wasserstein distance
        fig_wasser = go.Figure(go.Indicator(
            mode="gauge+number",
            value=distance,
            title={"text": "Wasserstein Distance"},
            gauge={
                "axis": {"range": [None, max_range]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, max_range*0.5], "color": "lightgray"},
                    {"range": [max_range*0.5, max_range], "color": "gray"}
                ]
            }
        ))
        st.plotly_chart(fig_wasser)

elif equation_type == "Wasserstein Distance":
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    # Contents of the first column
    with col1:
        st.subheader("Matrix A")
        # Sliders for setting the covariance matrix elements of matrix A
        A_xx = st.slider("A[0,0]", 0.1, 3.0, 1.0, 0.1)
        A_pp = st.slider("A[1,1]", 0.1, 3.0, 1.0, 0.1)
        A_xp = st.slider("A[0,1] and A[1,0]", -1.0, 1.0, 0.0, 0.1)

        st.subheader("Matrix B")
        # Sliders for setting the covariance matrix elements of matrix B
        B_xx = st.slider("B[0,0]", 0.1, 3.0, 1.0, 0.1)
        B_pp = st.slider("B[1,1]", 0.1, 3.0, 1.0, 0.1)
        B_xp = st.slider("B[0,1] and B[1,0]", -1.0, 1.0, 0.0, 0.1)

    # Contents of the second column
    with col2:
        # Construct matrices A and B from the slider values
        matrix_A = np.array([[A_xx, A_xp], [A_xp, A_pp]])
        matrix_B = np.array([[B_xx, B_xp], [B_xp, B_pp]])

        # Display matrices A and B
        st.subheader("Matrix A")
        st.write(matrix_A)

        st.subheader("Matrix B")
        st.write(matrix_B)

        # Calculate the Wasserstein distance between matrices A and B
        distance = wasserstein_distance(matrix_A, matrix_B)

        # Determine a dynamic maximum range for the gauge based on the computed distance
        max_range = distance * 1.2 if distance > 0 else 1.0

        # Create an improved interactive gauge indicator for Wasserstein Distance
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=distance,
            title={"text": "Wasserstein Distance"},
            gauge={
                "axis": {"range": [None, max_range]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, max_range*0.5], "color": "lightgray"},
                    {"range": [max_range*0.5, max_range], "color": "gray"}
                ]
            }
        ))
        st.plotly_chart(fig)
