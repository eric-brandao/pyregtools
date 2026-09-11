Installation
============

Requirements
------------

PyRegTools requires Python 3.12 or newer.

It is recommended to install PyRegTools in a dedicated Python
environment to avoid conflicts with other packages.

Creating a Conda environment
----------------------------

If you use Anaconda or Miniconda, create a new environment with:

.. code-block:: bash

   conda create -n pyregtools python=3.12

Activate the environment with:

.. code-block:: bash

   conda activate pyregtools

Installing PyRegTools
---------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/eric-brandao/pyregtools.git
   cd pyregtools

Then install PyRegTools and its dependencies:

.. code-block:: bash

   python -m pip install .

Development installation
------------------------

If you intend to modify PyRegTools, install it in editable mode together
with the testing and documentation dependencies:

.. code-block:: bash

   python -m pip install -e ".[test,docs]"