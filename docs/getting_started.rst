Getting Started
===============

Mathematical models and data play an important role in modern science and engineering. 
The field of inverse problems bridges these two domains and studies whether and how to
infer model parameters from measurements that are often incomplete, sparse, and/or 
contaminated by noise.

PyRegTools provides tools for solving discrete inverse problems. A
linear discrete inverse problem can be written as

.. math::

   \mathbf{A}\mathbf{x} = \mathbf{b},

where :math:`\mathbf{A} \in \mathbb{C}^{M \times L}` is the forward
model, :math:`\mathbf{x} \in \mathbb{C}^{L}` is the unknown solution,
and :math:`\mathbf{b} \in \mathbb{C}^{M}` contains the measured data 
(typically with some noise).

Inverse problems are often ill-conditioned, meaning that small
perturbations in the measured data can produce large changes in the
estimated solution. Regularization methods introduce additional
information or constraints to stabilize the solution.

For example, the Tikhonov-regularized solution is obtained by solving

.. math::

   \mathbf{x}_{\lambda}
   =
   \underset{\mathbf{x}}{\operatorname{argmin}}
   \left\{
   \|\mathbf{A}\mathbf{x}-\mathbf{b}\|_2^2
   +
   \lambda^2\|\mathbf{x}\|_2^2
   \right\},

where :math:`\lambda` is the regularization parameter. 

There are more details and functionalities that PyRegTools can handle, but it is 
possible to explore them later via examples.

A typical workflow consists of:

1. gathering measurement data,
2. defining a mathematical model in matrix form,
3. selecting a regularization parameter or strategy,
4. computing a regularized solution,
5. evaluating the solution and reconstructed quantities.

So, PyRegTools is primarily concerned with steps 3 and 4. For that, there are

- regularization solvers based on Tikhonov regularization, Truncated Singular Value Decomposition, etc,
- iterative solvers,
- automatic regularization parameter choice strategies (L-curve, Generalized Cross Validation, and more),
- solvers considering sparsity,
- toy problems and notebook examples for some theory and practice.

Steps 1 and 2 are addressed by toy problems in PyRegTools and step 5 is somewhat easy to handle.

The theory and code are based on the book: Discrete Inverse Problems - insight and algorithms, 
Per Christian Hansen, Technical University of Denmark, DTU compute, 2010. 
Several routines in PyRegTools are based on or compared against 
`Regularization Tools <https://www.imm.dtu.dk/~pcha/Regutools/>`_.


Detailed examples
-----------------

Complete examples are provided as Jupyter notebooks in the
``notebooks`` directory of the repository. These notebooks demonstrate
the full workflow, including problem generation, parameter selection,
regularized solution, and comparison with reference solutions.