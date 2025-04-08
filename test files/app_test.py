import cmath
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.linalg import sqrtm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

st.set_page_config(page_title="Plot Your Equation", layout="wide")

st.title("Explore Your Functions Interactively")

# Sidebar for controls
st.sidebar.header("Parameters")
equation_type = st.sidebar.selectbox(
    "Select Your Eqaution",
    ["Wigner Function (n)", "Gaussian Wigner Function",
     "Coupling Matrix", "Gaussian Wigner Function 4D"]
)


# Defining functions
def wigner_function(n, x, p):
    """
    EN: Calculates the Wigner function for a quantum harmonic oscillator using Laguerre polynomials. 
    This function takes the quantum number n and two phase space variables (position x and momentum p), 
    and returns the Wigner distribution value at each (x, p) point.

    FR : Calcule la fonction de Wigner pour un oscillateur harmonique quantique en utilisant les polynômes de Laguerre. 
    Cette fonction prend en entrée le nombre quantique n ainsi que deux variables d’espace de phase (position x et quantité de mouvement p), 
    et retourne la valeur de la distribution de Wigner pour chaque point (x, p).
    """
    prefactor = (-1)**n / np.pi
    exponent = np.exp(-x**2 - p**2)
    laguerre_term = sp.eval_genlaguerre(n, 0, 2 * (x**2 + p**2))
    return prefactor * exponent * laguerre_term


# EN: Calculates the 4D Gaussian Wigner function for a bipartite quantum system using a covariance matrix.
# FR : Calcule la fonction de Wigner gaussienne 4D pour un système quantique biparti en utilisant une matrice de covariance.
def wigner_gaussian4D(x1, p1, x2, p2, gamma):
    """
    EN: Computes the 4D Gaussian Wigner function for a bipartite (two-mode) quantum system characterized by a 4x4 covariance matrix gamma. 
    The function evaluates the Wigner function at a specific point (x1, p1, x2, p2) in the four-dimensional phase space, 
    assuming the system satisfies the quantum uncertainty principle.

    FR : Calcule la fonction de Wigner gaussienne 4D pour un système quantique biparti (à deux modes) à partir d’une matrice de covariance 4x4 gamma. 
    La fonction évalue la fonction de Wigner à un point spécifique (x1, p1, x2, p2) de l’espace de phase à quatre dimensions, 
    en supposant que le système satisfait au principe d’incertitude quantique.
    """
    if uncertainty(gamma):
        # Create vector r
        vec_r = np.array([[x1, p1, x2, p2]]).reshape(-1, 1)

        # Transpose of vec_r
        vec_r_t = np.transpose(vec_r)

        # Determinant and inverse of gamma
        det = np.linalg.det(gamma)
        inv = np.linalg.inv(gamma)

        norm = 1 / ((2 * np.pi)**2 * np.sqrt(det))
        exponent = -0.5 * vec_r_t @ inv @ vec_r

        print(f"vec_r T: {vec_r_t.flatten()}")
        print("\n")
        print(f"vec_r: {vec_r}")
        print("\n")

        return norm * np.exp(exponent)
    else:
        return "Gamma doesn't respect uncertainty relation!"


# EN: Calculates the 2D Gaussian Wigner function from a point r and a 2x2 covariance matrix.
# FR : Calcule la fonction de Wigner gaussienne 2D à partir d’un point r et d’une matrice de covariance 2x2.
def wigner_gaussian(r, covar_mx):
    """
    EN: Computes the value of the 2D Gaussian Wigner function at a given point r = (x, p), 
    using a 2x2 covariance matrix that describes the quantum state's spread and correlation in phase space. 
    This is used to model single-mode Gaussian states.

    FR : Calcule la valeur de la fonction de Wigner gaussienne 2D en un point donné r = (x, p), 
    en utilisant une matrice de covariance 2x2 décrivant l’étendue et les corrélations de l’état quantique dans l’espace de phase. 
    Cette fonction est utilisée pour modéliser des états gaussiens à un seul mode.
    """
    det = np.linalg.det(covar_mx)
    inv = np.linalg.inv(covar_mx)
    norm = 1 / (2 * np.pi * np.sqrt(det))
    exponent = -0.5 * r @ inv @ r
    return norm * np.exp(exponent)


# EN: Checks if the given covariance matrix gamma satisfies the uncertainty principle (Heisenberg's inequality).
# FR : Vérifie si la matrice de covariance gamma satisfait le principe d’incertitude (inégalité de Heisenberg).
def uncertainty(gamma):
    """
    EN: Checks whether the input covariance matrix gamma respects the generalized uncertainty principle 
    by computing the matrix M = gamma + (i/2) * Omega and verifying if it is Hermitian and positive semi-definite. 
    This ensures that the quantum state represented by gamma is physically valid.

    FR : Vérifie si la matrice de covariance gamma respecte le principe d’incertitude généralisé 
    en calculant la matrice M = gamma + (i/2) * Omega et en vérifiant si elle est hermitienne et semi-définie positive. 
    Cela garantit que l’état quantique représenté par gamma est physiquement valide.
    """
    gamma = np.array(gamma, dtype=complex)
    dim = gamma.shape[0]

    # Must be 2n x 2n
    if gamma.shape[1] != dim or (dim % 2 != 0):
        return False

    n = dim // 2

    # Block-diagonal assembly
    Omega = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0]
    ], dtype=complex)

    # print(f"omega is :\n {Omega}")

    # Form M = gamma + (i/2)*Omega
    M = gamma + 1j/2 * Omega
    # print(f"M is :\n {M}")

    # Check Hermiticity (M should equal its own conjugate transpose)
    if not np.allclose(M, M.conj().T):
        return False

    # Check that M is positive semidefinite (all eigenvalues >= 0)
    eigenvals = np.linalg.eigvalsh(M)  # eigvalsh is for Hermitian matrices
    # Allow a tiny numerical tolerance
    return np.all(eigenvals > -1e-14)


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
    # Step 1: Compute the positive square root of A
    sqrtA = sqrtm(A)

    # Step 2: Compute X = sqrt( sqrt(A) * B * sqrt(A) )
    X = sqrtm(sqrtA @ B @ sqrtA)

    # Clean up any tiny imaginary parts due to numerical precision
    X = np.real_if_close(X)

    return X


st.sidebar.markdown("---")
st.sidebar.subheader("Equation Info")

if equation_type == "Wigner Function (n)":
    st.sidebar.latex(
        r"W_n(x,p) = \frac{(-1)^n}{\pi} e^{-(x^2+p^2)} L_n(2(x^2+p^2))")
    st.sidebar.write("where Lₙ is the Laguerre polynomial of degree n")

elif equation_type == "Gaussian Wigner Function":
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

# Add custom equation input option
st.sidebar.markdown("---")
st.sidebar.subheader("Custom Equation")
st.sidebar.write("Coming soon: Add your own equation")

# Instructions
st.sidebar.markdown("---")
with st.sidebar.expander("Instructions"):
    st.write("""
    1. Select an equation type from the dropdown
    2. Adjust parameters using the sliders
    3. View the resulting plot in real-time
    4. Change plot type or colormap as needed
    5. For Wigner Function, you can download the data as CSV
    """)

# Main content area
if equation_type == "Wigner Function (n)":
    col1, col2 = st.columns([1, 2])  # we create 2 cols

    # contents of col1
    with col1:
        # Quant. Num. slider
        # Starts from 0 to 20, with default at 5
        n = st.slider("Quantum Number (n)", 0, 20, 5)

        # Slider for resolution
        # Starts from 20 to 200, with default at 100
        resolution = st.slider("Resolution", 20, 200, 100)

        # X Axes Range Slider
        # Range goes from -5.0 to 5.0 with default from -3.0 to 3.0
        x_axes_range = st.slider("X Range", -5.0, 5.0, (-3.0, 3.0))

        # Y Axes Range Slider
        # Range goes from -5.0 to 5.0 with default from -3.0 to 3.0
        p_axes_range = st.slider("P Range", -5.0, 5.0, (-3.0, 3.0))

        # Plot
        plot_type = st.radio("Plot Type", ["Surface", "Contour"])
        # Plot color map
        colormap = st.selectbox(
            "Color Map", ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])

    # Contents of col2
    with col2:
        # Define the grid (axes)
        # linspace(start, stop, value of the resolution slider)
        x_values = np.linspace(x_axes_range[0], x_axes_range[1], resolution)
        p_values = np.linspace(p_axes_range[0], p_axes_range[1], resolution)
        x_grid, p_grid = np.meshgrid(x_values, p_values)

        # Compute Wigner Function
        wigner_vals = wigner_function(n, x_grid, p_grid)

        # Plotting
        fig = plt.figure(figsize=(10, 8))

        if plot_type == "Surface":
            ax = fig.add_subplot(111, projection='3d')
            surface = ax.plot_surface(
                x_grid, p_grid, wigner_vals, cmap=colormap, edgecolor='none')
            ax.set_zlabel("$W_n(x, p)$")
        else:
            ax = fig.add_subplot(111)
            contour = ax.contourf(
                x_grid, p_grid, wigner_vals, cmap=colormap, levels=50)

        ax.set_xlabel("x")
        ax.set_ylabel("p")
        ax.set_title(f"Wigner Function for n = {n}")

        st.pyplot(fig)

        # Download data option
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

elif equation_type == "Gaussian Wigner Function":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Covariance Matrix")
        covar_xx = st.slider("Var(x)", 0.1, 3.0, 1.0, 0.1)
        covar_pp = st.slider("Var(p)", 0.1, 3.0, 1.0, 0.1)
        covar_xp = st.slider("Cov(x,p)", -1.0, 1.0, 0.0, 0.1)

        resolution = st.slider("Resolution", 20, 200, 100)
        x_range = st.slider("X Range", -5.0, 5.0, (-3.0, 3.0))
        p_range = st.slider("P Range", -5.0, 5.0, (-3.0, 3.0))

        plot_type = st.radio("Plot Type", ["Surface", "Contour"])
        colormap = st.selectbox(
            "Color Map", ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])

    with col2:
        # Construct covariance matrix
        covar_mx = np.array([[covar_xx, covar_xp], [covar_xp, covar_pp]])

        # Check if matrix is positive-definite (necessary for valid covariance matrix)
        if np.linalg.det(covar_mx) <= 0:
            st.warning(
                "Warning: Current values do not form a valid covariance matrix. Please adjust parameters.")

        # Display the matrix
        st.write("Covariance Matrix:")
        st.write(covar_mx)

        # Set up a grid over phase space (x and p)
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        p_vals = np.linspace(p_range[0], p_range[1], resolution)
        X, P = np.meshgrid(x_vals, p_vals)

        # Evaluate the Wigner function on the grid
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                r = np.array([X[i, j], P[i, j]])
                Z[i, j] = wigner_gaussian(r, covar_mx)

        # Create the plot
        fig = plt.figure(figsize=(10, 8))

        if plot_type == "Surface":
            ax = fig.add_subplot(111, projection='3d')
            surface = ax.plot_surface(X, P, Z, cmap=colormap, edgecolor='none')
            ax.set_zlabel("Wigner Function")
        else:
            ax = fig.add_subplot(111)
            contour = ax.contourf(X, P, Z, cmap=colormap, levels=50)
            plt.colorbar(contour)

        ax.set_xlabel("x")
        ax.set_ylabel("p")
        ax.set_title("Gaussian Wigner Function")

        st.pyplot(fig)

elif equation_type == "Coupling Matrix":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("State 1 Covariance Matrix")
        gamma1_xx = st.slider("γ1 Var(x)", 0.1, 3.0, 1.2, 0.1)
        gamma1_pp = st.slider("γ1 Var(p)", 0.1, 3.0, 1.1, 0.1)
        gamma1_xp = st.slider("γ1 Cov(x,p)", -1.0, 1.0, 0.5, 0.1)

        st.subheader("State 2 Covariance Matrix")
        gamma2_xx = st.slider("γ2 Var(x)", 0.1, 3.0, 1.0, 0.1)
        gamma2_pp = st.slider("γ2 Var(p)", 0.1, 3.0, 1.3, 0.1)
        gamma2_xp = st.slider("γ2 Cov(x,p)", -1.0, 1.0, 0.3, 0.1)

        st.subheader("Coupling Matrix")
        X_xx = st.slider("X(x1,x2)", -1.0, 1.0, 0.6, 0.1)
        X_pp = st.slider("X(p1,p2)", -1.0, 1.0, 0.7, 0.1)
        X_xp = st.slider("X(x1,p2)", -1.0, 1.0, 0.2, 0.1)
        X_px = st.slider("X(p1,x2)", -1.0, 1.0, 0.2, 0.1)

    with col2:
        # Define covariance matrices for two Gaussian states
        gamma_1 = np.array([[gamma1_xx, gamma1_xp], [gamma1_xp, gamma1_pp]])
        gamma_2 = np.array([[gamma2_xx, gamma2_xp], [gamma2_xp, gamma2_pp]])

        # Define optimal coupling matrix
        X_optimal = np.array([[X_xx, X_xp], [X_px, X_pp]])

        # Construct the full coupling matrix γ
        gamma = np.block([[gamma_1, X_optimal], [X_optimal.T, gamma_2]])

        # Verify that the matrix respects the uncertainty relation
        if uncertainty(gamma) == True:
            # Create X, Y grid for 3D plot
            X, Y = np.meshgrid(range(gamma.shape[0]), range(gamma.shape[1]))

            # Plot the 3D surface
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Surface plot
            ax.plot_surface(X, Y, gamma, cmap='coolwarm', edgecolor='k')

            # Labels and formatting
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(["Q1", "P1", "Q2", "P2"])
            ax.set_yticklabels(["Q1", "P1", "Q2", "P2"])
            ax.set_xlabel("Basis State 1")
            ax.set_ylabel("Basis State 2")
            ax.set_zlabel("Coupling Strength")
            ax.set_title(
                "3D Heatmap of the Coupling Matrix for Gaussian States")

            st.pyplot(fig)

        else:
            st.write("Your Matrix Doesn't Respect the Uncertainty Relation 1")

        # Display the numerical matrix
        st.write("Full Coupling Matrix:")
        st.write(pd.DataFrame(
            gamma,
            columns=["Q1", "P1", "Q2", "P2"],
            index=["Q1", "P1", "Q2", "P2"]
        ))


elif equation_type == "Gaussian Wigner Function 4D":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Vector r (x1, p1, x2, p2)")
        vecr_x1 = st.slider("x1", -5.0, 5.0, 0.0, 0.1)
        vecr_p1 = st.slider("p1", -5.0, 5.0, 0.0, 0.1)
        vecr_x2 = st.slider("x2", -5.0, 5.0, 0.0, 0.1)
        vecr_p2 = st.slider("p2", -5.0, 5.0, 0.0, 0.1)

        st.subheader("Covariance Matrix A (Top-Left, State A)")
        A_xx = st.slider("A[0,0] (Var x1)", 0.1, 3.0, 1.0, 0.1)
        A_pp = st.slider("A[1,1] (Var p1)", 0.1, 3.0, 1.0, 0.1)
        A_xp = st.slider("A[0,1] and A[1,0] (Cov x1,p1)", -1.0, 1.0, 0.0, 0.1)

        st.subheader("Covariance Matrix B (Bottom-Right, State B)")
        B_xx = st.slider("B[0,0] (Var x2)", 0.1, 3.0, 1.0, 0.1)
        B_pp = st.slider("B[1,1] (Var p2)", 0.1, 3.0, 1.0, 0.1)
        B_xp = st.slider("B[0,1] and B[1,0] (Cov x2,p2)", -1.0, 1.0, 0.0, 0.1)

        view_mode = st.radio("View Mode", ["2D Slice", "3D Plot", "4D Sweep"])
        st.subheader("Slice Selection")
        slice_axes = st.selectbox("Select 2D slice to plot", [
            ("x1", "p1"),
            ("x1", "x2"),
            ("x1", "p2"),
            ("p1", "x2"),
            ("p1", "p2"),
            ("x2", "p2")
        ])

        colormap = st.selectbox(
            "Color Map", ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])

    with col2:
        vector_r = np.array([[vecr_x1], [vecr_p1], [vecr_x2], [vecr_p2]])

        matrix_A = np.array([[A_xx, A_xp], [A_xp, A_pp]])
        matrix_B = np.array([[B_xx, B_xp], [B_xp, B_pp]])
        zero_block = np.zeros((2, 2))
        gamma = np.block([[matrix_A, zero_block], [zero_block, matrix_B]])

        st.subheader("Current Vector r")
        st.dataframe(pd.DataFrame(vector_r, index=[
                     "x1", "p1", "x2", "p2"], columns=["Value"]))

        st.subheader("Current Covariance Matrix γ")
        st.dataframe(pd.DataFrame(gamma, columns=[
                     "x1", "p1", "x2", "p2"], index=["x1", "p1", "x2", "p2"]))

        result = wigner_gaussian4D(vecr_x1, vecr_p1, vecr_x2, vecr_p2, gamma)
        st.subheader("Wigner Function Value at r")
        st.write(result)

        if view_mode == "2D Slice":
            st.subheader(
                f"Wigner Function Slice: {slice_axes[0]} vs {slice_axes[1]}")
            resolution = 100
            axis_vals = np.linspace(-3, 3, resolution)
            X, Y = np.meshgrid(axis_vals, axis_vals)
            Z = np.zeros_like(X)

            for i in range(resolution):
                for j in range(resolution):
                    coords = {"x1": vecr_x1, "p1": vecr_p1,
                              "x2": vecr_x2, "p2": vecr_p2}
                    coords[slice_axes[0]] = X[i, j]
                    coords[slice_axes[1]] = Y[i, j]
                    Z[i, j] = wigner_gaussian4D(
                        coords["x1"], coords["p1"], coords["x2"], coords["p2"], gamma
                    )

            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(X, Y, Z, cmap=colormap, levels=50)
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel(slice_axes[0])
            ax.set_ylabel(slice_axes[1])
            st.pyplot(fig)

        elif view_mode == "3D Plot":
            st.subheader(
                f"Wigner Function 3D Surface: {slice_axes[0]} vs {slice_axes[1]}")
            resolution = 100
            axis_vals = np.linspace(-3, 3, resolution)
            X, Y = np.meshgrid(axis_vals, axis_vals)
            Z = np.zeros_like(X)

            for i in range(resolution):
                for j in range(resolution):
                    coords = {"x1": vecr_x1, "p1": vecr_p1,
                              "x2": vecr_x2, "p2": vecr_p2}
                    coords[slice_axes[0]] = X[i, j]
                    coords[slice_axes[1]] = Y[i, j]
                    Z[i, j] = wigner_gaussian4D(
                        coords["x1"], coords["p1"], coords["x2"], coords["p2"], gamma
                    )

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap=colormap, edgecolor='none')
            ax.set_xlabel(slice_axes[0])
            ax.set_ylabel(slice_axes[1])
            ax.set_zlabel("Wigner Value")
            st.pyplot(fig)

        elif view_mode == "4D Sweep":
            st.subheader("4D Visualization Grid")
            sweep_axis = st.selectbox("Sweep Across", ["x1", "p1", "x2", "p2"])
            fixed_axis = st.selectbox("Fix Axis", [a for a in [
                                      "x1", "p1", "x2", "p2"] if a not in slice_axes and a != sweep_axis])
            fixed_value = st.slider(f"Fixed {fixed_axis}", -5.0, 5.0, 0.0, 0.1)

            sweep_vals = np.linspace(-2, 2, 5)
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))

            for idx, val in enumerate(sweep_vals):
                coords = {"x1": vecr_x1, "p1": vecr_p1,
                          "x2": vecr_x2, "p2": vecr_p2}
                coords[fixed_axis] = fixed_value
                x_vals = np.linspace(-3, 3, 80)
                y_vals = np.linspace(-3, 3, 80)
                X, Y = np.meshgrid(x_vals, y_vals)
                Z = np.zeros_like(X)

                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        coords[slice_axes[0]] = X[i, j]
                        coords[slice_axes[1]] = Y[i, j]
                        coords[sweep_axis] = val
                        Z[i, j] = wigner_gaussian4D(
                            coords["x1"], coords["p1"], coords["x2"], coords["p2"], gamma
                        )

                cs = axes[idx].contourf(X, Y, Z, levels=30, cmap=colormap)
                axes[idx].set_title(f"{sweep_axis} = {val:.2f}")
                axes[idx].set_xlabel(slice_axes[0])
                axes[idx].set_ylabel(slice_axes[1])

            plt.tight_layout()
            st.pyplot(fig)
