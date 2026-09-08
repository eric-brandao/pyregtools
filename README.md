# pyregtools

**pyregtools** is a python package used for tasks related to solving regularized inverse problems. The main applications are solving inverse problems with Tikhonov regularization, but other methods are also possible. The package contains functionality for:

- authomatic choice of the regularization parameter via: the L-curve criterion, the discrepancy principle, Generatized Cross Validation (GCV), and Normalized Cumulative Periodogram (NCP);
- iterative solvers such as: Landweber and Cimmino, Algebraic Reconstruction Technique (ART), and Conjugate Gradient Least Squares (CGLS)

**pyregtools** also contains a series of toy problems, that can be used both for testing the regularization routines, and for teaching and learning about inverse problems. Among this test problems are:

- the Gravity surveying problem: estimate density based on measured acceleration
- the Heat problem: estimate the temperature over time at a hot spot based on a remote measurement.
- the Sky problem: image debluring of a sparse set of stars on a dark sky
- image deblurring for photgraphy
- the 2D Plane-Wave expansion problem: estimate amplitude and phase of plane waves based on microphone array measurements

other test problems might be implemented and contributions are welcome

The routines are largely based on the book: Discrete Inverse Problems Insight and Algorithms, SIAM, 2010, by Per Christian Hansen (professor of the Technical University of Denmark - DTU compute). The book link can be find at http://www.imm.dtu.dk/~pcha/DIPbook.html

The routines are tested against the original implementation on matlab (http://www.imm.dtu.dk/~pcha/Regutools/)



