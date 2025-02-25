# Interactive Function Explorer

An interactive application for exploring and visualizing quantum Wigner functions and related quantum state representations.

## Overview

This tool allows researchers and students to interactively explore different quantum state representations in phase space:

- **Wigner Functions**: Visualize quantum states with quantum number n
- **Gaussian Wigner Functions**: Explore Gaussian states with adjustable covariance parameters
- **Coupling Matrices**: Visualize the correlation between two Gaussian quantum states

All visualizations are interactive, allowing for real-time parameter adjustments and different viewing options.

## Requirements

- Python 3.7 or higher

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Hossein-Hmb/Interactive_Fucntion_Explorer.git && cd Interactive_Fucntion_Explorer
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

After installation, simply run:

```bash
streamlit run app.py
```

The application will start and automatically open in your default web browser.

## Features

- Interactive sliders to adjust quantum parameters
- Multiple visualization types (3D surface, contour plots)
- Adjustable plot resolution and ranges
- Color scheme selection
- Mathematical equation display
- Data export capabilities

## Usage Tips

- Use the sidebar to select different equation types and adjust parameters
- Try both surface and contour plot types to view the data differently
- For better performance with complex calculations, reduce the resolution
- Hover over plot points to see exact values

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
