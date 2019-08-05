import GPUifyLoops: @launch, @loop, @unroll

using Oceananigans.TurbulenceClosures

const coordinates = (:x, :y, :z)
const dims = length(coordinates)
const solution_fields = (:u, :v, :w, :T, :S)
const nsolution = length(solution_fields)

abstract type BCType end
struct Periodic <: BCType end
struct Flux <: BCType end
struct Gradient <: BCType end
struct Value <: BCType end

"""
    BoundaryCondition(BCType, condition)

Construct a boundary condition of `BCType` with `condition`,
where `BCType` is `Flux` or `Gradient`. `condition` may be a
number, array, or a function with signature:

    `condition(i, j, grid, t, iteration, U, Φ) = # function definition`

where `i` and `j` are indices along the boundary.
"""
struct BoundaryCondition{C<:BCType, T}
    condition :: T
end

bctype(bc::BoundaryCondition{C}) where C = C

# Constructors
BoundaryCondition(Tbc, c) = BoundaryCondition{Tbc, typeof(c)}(c)

# Adapt boundary condition struct to be GPU friendly and passable to GPU kernels.
Adapt.adapt_structure(to, b::BoundaryCondition{C, A}) where {C<:BCType, A<:AbstractArray} =
    BoundaryCondition(C, Adapt.adapt(to, parent(b.condition)))

"""
    CoordinateBoundaryConditions

Construct `CoordinateBoundaryConditions` to be applied along coordinate `c`, where
`c` is `:x`, `:y`, or `:z`. A CoordinateBoundaryCondition has two fields
`left` and `right` that store boundary conditions on the 'left' (negative side)
and 'right' (positive side) of a given coordinate.
"""
mutable struct CoordinateBoundaryConditions{L, R}
     left :: L
    right :: R
end

function CoordinateBoundaryConditions(;
     left = BoundaryCondition(Periodic, nothing),
    right = BoundaryCondition(Periodic, nothing)
   )
    return CoordinateBoundaryConditions(left, right)
end

PeriodicBoundaryConditions() =
    CoordinateBoundaryConditions(BoundaryCondition(Periodic, nothing),
                                 BoundaryCondition(Periodic, nothing))

"""
    ZBoundaryConditions(top=BoundaryCondition(Periodic, nothing),
                        bottom=BoundaryCondition(Periodic, nothing))

Returns `CoordinateBoundaryConditions` with specified `top`
and `bottom` boundary conditions.
"""
function ZBoundaryConditions(;
       top = BoundaryCondition(Flux, 0),
    bottom = BoundaryCondition(Flux, 0)
   )
    return CoordinateBoundaryConditions(top, bottom)
end

const CBC = CoordinateBoundaryConditions

#=
Here we overload setproperty! and getproperty to permit users to call
the 'right' and 'left' bcs in the z-direction 'bottom' and 'top'.

Note that 'right' technically corresponds to face point N+1. Thus
the fact that right == bottom is associated with the reverse z-indexing
convention. With ordinary indexing, right == top.
=#
Base.setproperty!(cbc::CBC, side::Symbol, bc) = setbc!(cbc, Val(side), bc)
setbc!(cbc::CBC, ::Val{S}, bc) where S = setfield!(cbc, S, bc)
setbc!(cbc::CBC, ::Val{:bottom}, bc) = setfield!(cbc, :right, bc)
setbc!(cbc::CBC, ::Val{:top}, bc) = setfield!(cbc, :left, bc)

Base.getproperty(cbc::CBC, side::Symbol) = getbc(cbc, Val(side))
getbc(cbc::CBC, ::Val{S}) where S = getfield(cbc, S)
getbc(cbc::CBC, ::Val{:bottom}) = getfield(cbc, :right)
getbc(cbc::CBC, ::Val{:top}) = getfield(cbc, :left)

"""
    FieldBoundaryConditions <: FieldVector{dims, CoordinateBoundaryConditions}

Construct `FieldBoundaryConditions` for a field.
A FieldBoundaryCondition has `CoordinateBoundaryConditions` in
`x`, `y`, and `z`.
"""
struct FieldBoundaryConditions{X, Y, Z}
    x :: X
    y :: Y
    z :: Z
end

function FieldBoundaryConditions(;
    x = CoordinateBoundaryConditions(),
    y = CoordinateBoundaryConditions(),
    z = CoordinateBoundaryConditions()
    )
    return FieldBoundaryConditions(x, y, z)
end

function HorizontallyPeriodicBCs(;    top = BoundaryCondition(Flux, 0),
                                   bottom = BoundaryCondition(Flux, 0)
                                )

    x = PeriodicBoundaryConditions()
    y = PeriodicBoundaryConditions()
    z = CoordinateBoundaryConditions(top, bottom)

    return FieldBoundaryConditions(x, y, z)
end

function ChannelBCs(;  north = BoundaryCondition(Flux, 0),
                       south = BoundaryCondition(Flux, 0),
                         top = BoundaryCondition(Flux, 0),
                      bottom = BoundaryCondition(Flux, 0)
                    )

    x = PeriodicBoundaryConditions()
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(top, bottom)

    return FieldBoundaryConditions(x, y, z)
end

struct ModelBoundaryConditions{UBC, VBC, WBC, TBC, SBC}
    u :: UBC
    v :: VBC
    w :: WBC
    T :: TBC
    S :: SBC
end

# sensible alias
const BoundaryConditions = ModelBoundaryConditions

"""
    ModelBoundaryConditions(u=u_boundary_conditions, ...)

Returns model boundary conditions for `u`, `v`, `w`, `T`, and `S`.
"""
function ModelBoundaryConditions(;
    u = HorizontallyPeriodicBCs(),
    v = HorizontallyPeriodicBCs(),
    w = HorizontallyPeriodicBCs(),
    T = HorizontallyPeriodicBCs(),
    S = HorizontallyPeriodicBCs()
   )
    return ModelBoundaryConditions(u, v, w, T, S)
end

function ChannelModelBoundaryConditions(;
    u = ChannelBCs(),
    v = ChannelBCs(),
    w = ChannelBCs(),
    T = ChannelBCs(),
    S = ChannelBCs()
   )
    return ModelBoundaryConditions(u, v, w, T, S)
end

#
# Some helper functions for constructing boundary conditions
#

"""
    BoundaryConditions(u=FieldBoundaryConditions(), ...)

    BoundaryConditions(fld, coord, side, bc)

Return an instance of `ModelBoundaryConditions` with one non-doubly-periodic
boundary condition `bc` on `fld` along coordinate `coord` at `side`.
"""
function ModelBoundaryConditions(fld, coord, ::Val{S}, bc) where S
    coordbcs = CoordinateBoundaryConditions(; Dict((S => bc))...)
      fldbcs = FieldBoundaryConditions(; Dict((coord => coordbcs))...)
    modelbcs = ModelBoundaryConditions(; Dict((fld => fldbcs))...)
    return modelbcs
end

# Alias 'right' and 'left' to 'bottom' and 'top' to clarify setting
# z boundary conditions.
ModelBoundaryConditions(fld, coord, s::Symbol, bc) =
    ModelBoundaryConditions(fld, coord, Val(s), bc)

ModelBoundaryConditions(fld, coord, ::Val{:bottom}, bc) =
    ModelBoundaryConditions(fld, coord, Val(:right), bc)

ModelBoundaryConditions(fld, coord, ::Val{:top}, bc) =
    ModelBoundaryConditions(fld, coord, Val(:left), bc)

#=
Notes:

- We assume that the boundary tendency has been previously calculated for
  a 'no-flux' boundary condition.

  This means that boudnary conditions take the form of an addition/subtraction
  to the tendency associated with a flux at point `aaf` below the bottom cell.
  This paradigm holds as long as consider boundary conditions on `aaf`
  variables only, where a is "any" of c or f. See the src/operators/README for
  more information on the naming convention for different grid point locations.

 - We use the physics-based convention that

        flux = -κ * gradient,

    and that

        tendency = ∂c/∂t = Gc = - ∇ ⋅ flux
=#

const BC = BoundaryCondition
const FBCs = FieldBoundaryConditions

# Do nothing in cases not explicitly defined.
# These functions are called in cases where one of the
# z-boundaries is set, but not the other.
@inline apply_z_top_bc!(args...) = nothing
@inline apply_z_bottom_bc!(args...) = nothing

# These functions compute vertical fluxes for (A, A, C) quantities. They are not currently used.
@inline ∇κ∇c_t(κ, ct, ct₋₁, flux, ΔzC, ΔzF) = (      -flux        - κ*(ct - ct₋₁)/ΔzC ) / ΔzF
@inline ∇κ∇c_b(κ, cb, cb₊₁, flux, ΔzC, ΔzF) = ( κ*(cb₊₁ - cb)/ΔzC +       flux        ) / ΔzF

# Multiple dispatch on the type of boundary condition
getbc(bc::BC{C, <:Number}, args...)              where C = bc.condition
getbc(bc::BC{C, <:AbstractArray}, i, j, args...) where C = bc.condition[i, j]
getbc(bc::BC{C, <:Function}, args...)            where C = bc.condition(args...)

Base.getindex(bc::BC{C, <:AbstractArray}, inds...) where C = getindex(bc.condition, inds...)

"""
    apply_z_top_bc!(top_bc, i, j, grid, c, Gc, κ, t, iteration, U, Φ)

Add the part of flux divergence associated with a top boundary condition on c.
to Gc, where the conservation equation for c is ∂c/∂t = Gc.
If `top_bc.condition` is a function, the function must have the signature

    `top_bc.condition(i, j, grid, t, iteration, U, Φ)`

"""
@inline apply_z_top_bc!(top_flux::BC{<:Flux}, i, j, grid, c, Gc, κ, t, iteration, U, Φ) =
    Gc[i, j, 1] -= getbc(top_flux, i, j, grid, t, iteration, U, Φ) / grid.Δz

@inline apply_z_top_bc!(top_gradient::BC{<:Gradient}, i, j, grid, c, Gc, κ, t, iteration, U, Φ) =
    Gc[i, j, 1] += κ * getbc(top_gradient, i, j, grid, t, iteration, U, Φ) / grid.Δz

@inline apply_z_top_bc!(top_value::BC{<:Value}, i, j, grid, c, Gc, κ, t, iteration, U, Φ) =
    Gc[i, j, 1] += 2κ / grid.Δz^2 * (getbc(top_value, i, j, grid, t, iteration, U, Φ) - c[i, j, 1])

"""
    apply_z_bottom_bc!(bottom_bc, i, j, grid, c, Gc, κ, t, iteration, U, Φ)

Add the part of flux divergence associated with a bottom boundary condition on c.
to Gc, where the conservation equation for c is ∂c/∂t = Gc.
If `bottom_bc.condition` is a function, the function must have the signature

    `bottom_bc.condition(i, j, grid, t, iteration, U, Φ)`

"""
@inline apply_z_bottom_bc!(bottom_flux::BC{<:Flux}, i, j, grid, c, Gc, κ, t, iteration, U, Φ) =
    Gc[i, j, grid.Nz] += getbc(bottom_flux, i, j, grid, t, iteration, U, Φ) / grid.Δz

@inline apply_z_bottom_bc!(bottom_gradient::BC{<:Gradient}, i, j, grid, c, Gc, κ, t, iteration, U, Φ) =
    Gc[i, j, grid.Nz] -= κ * getbc(bottom_gradient, i, j, grid, t, iteration, U, Φ) / grid.Δz

@inline apply_z_bottom_bc!(bottom_value::BC{<:Value}, i, j, grid, c, Gc, κ, t, iteration, U, Φ) =
    Gc[i, j, grid.Nz] -= 2κ / grid.Δz^2 * (c[i, j, grid.Nz] - getbc(bottom_value, i, j, grid, t, iteration, U, Φ))

# Do nothing if both left and right boundary conditions are periodic.
apply_bcs!(::CPU, ::Val{:x}, Bx, By, Bz,
    left_bc::BC{<:Periodic, T}, right_bc::BC{<:Periodic, T}, args...) where {T} = nothing
apply_bcs!(::CPU, ::Val{:y}, Bx, By, Bz,
    left_bc::BC{<:Periodic, T}, right_bc::BC{<:Periodic, T}, args...) where {T} = nothing
apply_bcs!(::CPU, ::Val{:z}, Bx, By, Bz,
    left_bc::BC{<:Periodic, T}, right_bc::BC{<:Periodic, T}, args...) where {T} = nothing

apply_bcs!(::GPU, ::Val{:x}, Bx, By, Bz,
    left_bc::BC{<:Periodic, T}, right_bc::BC{<:Periodic, T}, args...) where {T} = nothing
apply_bcs!(::GPU, ::Val{:y}, Bx, By, Bz,
    left_bc::BC{<:Periodic, T}, right_bc::BC{<:Periodic, T}, args...) where {T} = nothing
apply_bcs!(::GPU, ::Val{:z}, Bx, By, Bz,
    left_bc::BC{<:Periodic, T}, right_bc::BC{<:Periodic, T}, args...) where {T} = nothing

# First, dispatch on coordinate.
apply_bcs!(arch, ::Val{:x}, Bx, By, Bz, args...) =
    @launch device(arch) threads=(Tx, Ty) blocks=(By, Bz) apply_x_bcs!(args...)
apply_bcs!(arch, ::Val{:y}, Bx, By, Bz, args...) =
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, Bz) apply_y_bcs!(args...)
apply_bcs!(arch, ::Val{:z}, Bx, By, Bz, args...) =
    @launch device(arch) threads=(Tx, Ty) blocks=(Bx, By) apply_z_bcs!(args...)

@inline get_top_κ(κ::Number, args...) = κ
@inline get_bottom_κ(κ::Number, args...) = κ

@inline get_top_κ(κ::AbstractArray, i, j, args...) = κ[i, j, 1]
@inline get_bottom_κ(κ::AbstractArray, i, j, grid, args...) = κ[i, j, grid.Nz]

# ConstantSmagorinsky does not compute or store κ so we will compute κ = ν / Pr.
@inline get_top_κ(ν::AbstractArray, i, j, grid, closure::ConstantSmagorinsky, args...) = ν[i, j, 1] / closure.Pr
@inline get_bottom_κ(ν::AbstractArray, i, j, grid, closure::ConstantSmagorinsky, args...) = ν[i, j, grid.Nz] / closure.Pr

"""
    apply_z_bcs!(top_bc, bottom_bc, grid, c, Gc, κ, closure, eos, g, t, iteration, U, Φ)

Apply a top and/or bottom boundary condition to variable c. Note that this kernel
must be launched on the GPU with blocks=(Bx, By). If launched with blocks=(Bx, By, Bz),
the boundary condition will be applied Bz times!
"""
function apply_z_bcs!(top_bc, bottom_bc, grid, c, Gc, κ, closure, eos, g, t, iteration, U, Φ)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            κ_top = get_top_κ(κ, i, j, grid, closure, eos, g, U, Φ)
            κ_bottom = get_bottom_κ(κ, i, j, grid, closure, eos, g, U, Φ)

               apply_z_top_bc!(top_bc,    i, j, grid, c, Gc, κ_top, t, iteration, U, Φ)
            apply_z_bottom_bc!(bottom_bc, i, j, grid, c, Gc, κ_bottom, t, iteration, U, Φ)
        end
    end
end
