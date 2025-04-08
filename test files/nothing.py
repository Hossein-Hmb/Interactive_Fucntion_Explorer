elif equation_type == "Gaussian Wigner Function":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Vector r")
        vecr_x1 = st.slider("x1", -5.0, 5.0, 0.0, 0.1)
        vecr_p1 = st.slider("p1", -5.0, 5.0, 0.0, 0.1)
        vecr_x2 = st.slider("x2", -5.0, 5.0, 0.0, 0.1)
        vecr_p2 = st.slider("p2", -5.0, 5.0, 0.0, 0.1)

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
                Z[i, j] = wigner_gaussian(
                    vecr_x1, vecr_p1, vecr_x2, vecr_p2, covar_mx)

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

        # Display the numerical matrix
        st.write("Full Coupling Matrix:")
        st.write(pd.DataFrame(
            gamma,
            columns=["Q1", "P1", "Q2", "P2"],
            index=["Q1", "P1", "Q2", "P2"]
        ))

        # Display the vector r
        st.write("VEctor r:")
        st.write(pd.DataFrame(
            vector_r,
            columns=["Q1", "P1", "Q2", "P2"]
            # index=["Q1", "P1", "Q2", "P2"]
        ))
