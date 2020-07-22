# Coriolis

The Coriolis option determines whether the fluid experiences the effect of the Coriolis force, or rotation. Currently
three options are available: no rotation, $f$-plane, and $\beta$-plane.

!!! info "Coriolis vs. rotation"
    If you are wondering why this option is called "Coriolis" it is because rotational effects could include the
    Coriolis and centripetal forces, both of which arise in non-inertial reference frames. But here the model only
    considers the Coriolis force.

## No rotation

By default there is no rotation. This can be made explicit by passing `coriolis = nothing` to a model constructor.

## $f$-plane

To set up an $f$-plane with, for example, Coriolis parameter $f = 10^{-4} \text{s}^{-1}$

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest
julia> coriolis = FPlane(f=1e-4)
FPlane{Float64}: f = 1.00e-04
```

An $f$-plane can also be specified at some latitude on a spherical planet with a planetary rotation rate. For example,
to specify an $f$-plane at a latitude of $\varphi = 45°\text{N}$ on Earth which has a rotation rate of
$\Omega = 7.292115 \times 10^{-5} \text{s}^{-1}$

```jldoctest
julia> coriolis = FPlane(rotation_rate=7.292115e-5, latitude=45)
FPlane{Float64}: f = 1.03e-04
```

in which case the value of $f$ is given by $2\Omega\sin\varphi$.

To set up an $f$-plane with non-traditional Coriolis terms, for example, with
$\bm{f} = (0, f_y, f_z) = (0, 2, 1) \times 10^{-4} \text{s}^{-1}$,

```jldoctest
julia> coriolis = NonTraditionalFPlane(fz=1e-4, fy=2e-4)
NonTraditionalFPlane{Float64}: fz = 1.00e-04, fy = 2.00e-04
```

An $f$-plane with non-traditional Coriolis terms can also be specified at some latitude on a spherical planet
with a planetary rotation rate. For example, to specify an $f$-plane at a latitude of $\varphi = 45°\text{N}$
on Earth which has a rotation rate of $\Omega = 7.292115 \times 10^{-5} \text{s}^{-1}$

```jldoctest
julia> coriolis = NonTraditionalFPlane(rotation_rate=7.292115e-5, latitude=45)
NonTraditionalFPlane{Float64}: fz = 1.03e-04, fy = 1.03e-04
```

in which case $f_z = 2\Omega\sin\varphi$ and $f_y = 2\Omega\cos\varphi$.

## $\beta$-plane

To set up a $\beta$-plane the background rotation rate $f_0$ and the $\beta$ parameter must be specified. For example,
a $\beta$-plane with $f_0 = 10^{-4} \text{s}^{-1}$ and $\beta = 1.5 \times 10^{-11} \text{s}^{-1}\text{m}^{-1}$ can be
set up with

```jldoctest
julia> coriolis = BetaPlane(f₀=1e-4, β=1.5e-11)
BetaPlane{Float64}: f₀ = 1.00e-04, β = 1.50e-11
```

Alternatively, a $\beta$-plane can also be set up at some latitude on a spherical planet with a planetary rotation rate
and planetary radius. For example, to specify a $\beta$-plane at a latitude of $\varphi = 10\degree{S}$ on Earth
which has a rotation rate of $\Omega = 7.292115 \times 10^{-5} \text{s}^{-1}$ and a radius of $R = 6,371 \text{km}$

```jldoctest
julia> coriolis = BetaPlane(rotation_rate=7.292115e-5, latitude=-10, radius=6371e3)
BetaPlane{Float64}: f₀ = -2.53e-05, β = 2.25e-11
```

in which case $f_0 = 2\Omega\sin\varphi$ and $\beta = 2\Omega\cos\varphi / R$.
