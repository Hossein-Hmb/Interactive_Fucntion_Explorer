import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

st.set_page_config(page_title="Plot Your Equation", layout="wide")

st.title("Explore Your Functions Interactively")

# Sidebar for controls
st.sidebar.header("Parameters")
equation_type = st.sidebar.selectbox(
    "Select Your Eqaution",
    ["Wigner Function (n)", "Gaussian Wigner Function", "Coupling Matrix"]
)


# Defining functions
def wigner_function(n, x, p):
    prefactor = (-1)**n / np.pi
    exponent = np.exp(-x**2 - p**2)
    laguerre_term = sp.eval_genlaguerre(n, 0, 2 * (x**2 + p**2))
    return prefactor * exponent * laguerre_term


def wigner_gaussian(r, covar_mx):
    det = np.linalg.det(covar_mx)
    inv = np.linalg.inv(covar_mx)
    norm = 1 / (2 * np.pi * np.sqrt(det))
    exponent = -0.5 * r @ inv @ r
    return norm * np.exp(exponent)


# Add equation explanation
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
        ax.set_title("3D Heatmap of the Coupling Matrix for Gaussian States")

        st.pyplot(fig)

        # Display the numerical matrix
        st.write("Full Coupling Matrix:")
        st.write(pd.DataFrame(
            gamma,
            columns=["Q1", "P1", "Q2", "P2"],
            index=["Q1", "P1", "Q2", "P2"]
        ))
