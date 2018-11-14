# OceanDispatch.jl
A fast non-hydrostatic n-dimensional ocean model based on the MITgcm algorithm in Julia. The plan is to make it useful as a large eddy simulation (LES) model or as a fast super-parameterization to be embedded (or *dispatched*) within a global ocean model. As an embedded model it could resolve the sub-grid scale physics and communicate their effects back to the global model or act as a source of training data for statistical learning algorithms.

It may end up as a general-purpose ocean model that can be used in a hydrostatic or non-hydrostatic configuration with a friendly and intuitive user interface. Thanks high-level zero-cost abstractions in Julia we think we can make the model look and behave the same no matter the dimension or grid of the underlying simulation.
