# Coordinate system and notation

Oceananigans.jl is formulated in a Cartesian coordinate system ``\boldsymbol{x} = (x, y, z)`` 
with unit vectors ``\boldsymbol{\hat x}``, ``\boldsymbol{\hat y}``, and ``\boldsymbol{\hat z}``, 
where ``\boldsymbol{\hat x}`` points east, ``\boldsymbol{\hat y}`` points north, and ``\boldsymbol{\hat z}`` 
points 'upward', opposite the direction of gravitational acceleration.

We denote time with ``t``, partial derivatives with respect to time ``t`` or a coordinate ``x`` 
with ``\partial_t`` or ``\partial_x``, and denote the gradient operator ``\boldsymbol{\nabla} \equiv 
\partial_x \boldsymbol{\hat x} + \partial_y \boldsymbol{\hat y} + \partial_z \boldsymbol{\hat z}``. 
Horizontal gradients are denoted with ``\boldsymbol{\nabla}_h \equiv \partial_x \boldsymbol{\hat x} + \partial_y \boldsymbol{\hat y}``.

We use ``u``, ``v``, and ``w`` to denote the east, north, and vertical velocity components,
such that ``\boldsymbol{v} = u \boldsymbol{\hat x} + v \boldsymbol{\hat y} + w \boldsymbol{\hat z}``.
We reserve ``\boldsymbol{v}`` for the three-dimensional velocity field and use ``\boldsymbol{u}``
to denote the horizontal components of flow, i.e., ``\boldsymbol{u} = u \boldsymbol{\hat x} + 
v \boldsymbol{\hat y}``.