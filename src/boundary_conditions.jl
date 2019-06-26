import GPUifyLoops: @launch, @loop, @unroll, @synchronize

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

    `condition(i, j, grid, t, iteration, u, v, w, T, S) = # function definition`

where `i` and `j` are indices along the boundary.
"""
struct BoundaryCondition{C<:BCType, T}
    condition :: T
end

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
mutable struct CoordinateBoundaryConditions
     left :: BoundaryCondition
    right :: BoundaryCondition
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
struct FieldBoundaryConditions <: FieldVector{dims, CoordinateBoundaryConditions}
    x :: CoordinateBoundaryConditions
    y :: CoordinateBoundaryConditions
    z :: CoordinateBoundaryConditions
end

"""
    ModelBoundaryConditions <: FieldVector{nsolution, FieldBoundaryConditions}

Construct a boundary condition type full of default
`FieldBoundaryConditions` for u, v, w, T, S.
"""
struct ModelBoundaryConditions <: FieldVector{nsolution, FieldBoundaryConditions}
    u :: FieldBoundaryConditions
    v :: FieldBoundaryConditions
    w :: FieldBoundaryConditions
    T :: FieldBoundaryConditions
    S :: FieldBoundaryConditions
end

DoublyPeriodicBCs() = FieldBoundaryConditions(
                          CoordinateBoundaryConditions(
                              BoundaryCondition(Periodic, nothing),
                              BoundaryCondition(Periodic, nothing)),
                          CoordinateBoundaryConditions(
                              BoundaryCondition(Periodic, nothing),
                              BoundaryCondition(Periodic, nothing)),
                          CoordinateBoundaryConditions(
                              BoundaryCondition(Flux, 0),
                              BoundaryCondition(Flux, 0)))

ChannelBCs() = FieldBoundaryConditions(
                   CoordinateBoundaryConditions(
                       BoundaryCondition(Periodic, nothing),
                       BoundaryCondition(Periodic, nothing)),
                   CoordinateBoundaryConditions(
                       BoundaryCondition(Flux, 0),
                       BoundaryCondition(Flux, 0)),
                   CoordinateBoundaryConditions(
                       BoundaryCondition(Flux, 0),
                       BoundaryCondition(Flux, 0)))

"""
    ModelBoundaryConditions()

Return a default set of model boundary conditions. For now, this corresponds to a
doubly periodic domain, so `Periodic` boundary conditions along the x- and y-dimensions,
with no-flux boundary conditions at the top and bottom.
"""
function ModelBoundaryConditions()
    bcs = (DoublyPeriodicBCs() for i = 1:length(solution_fields))
    return ModelBoundaryConditions(bcs...)
end

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

        tendency = ∂ϕ/∂t = Gϕ = - ∇ ⋅ flux
=#

const BC = BoundaryCondition
const FBCs = FieldBoundaryConditions

# Do nothing in default case. These functions are called in cases where one of the
# z-boundaries is set, but not the other.
@inline apply_z_top_bc!(args...) = nothing
@inline apply_z_bottom_bc!(args...) = nothing

# These functions compute vertical fluxes for (A, A, C) quantities. They are not currently used.
@inline ∇κ∇ϕ_t(κ, ϕt, ϕt₋₁, flux, ΔzC, ΔzF) = (      -flux        - κ*(ϕt - ϕt₋₁)/ΔzC ) / ΔzF
@inline ∇κ∇ϕ_b(κ, ϕb, ϕb₊₁, flux, ΔzC, ΔzF) = ( κ*(ϕb₊₁ - ϕb)/ΔzC +       flux        ) / ΔzF

# Multiple dispatch on the type of boundary condition
getbc(bc::BC{C, <:Number}, args...)              where C = bc.condition
getbc(bc::BC{C, <:AbstractArray}, i, j, args...) where C = bc.condition[i, j]
getbc(bc::BC{C, <:Function}, args...)            where C = bc.condition(args...)

Base.getindex(bc::BC{C, <:AbstractArray}, inds...) where C = getindex(bc.condition, inds...)

"""
    apply_z_top_bc!(top_bc, i, j, grid, ϕ, Gϕ, κ, t, iteration, u, v, w, T, S)

Add the part of flux divergence associated with a top boundary condition on ϕ.
to Gϕ, where the conservation equation for ϕ is ∂ϕ/∂t = Gϕ.
If `top_bc.condition` is a function, the function must have the signature

    `top_bc.condition(i, j, grid, t, iteration, u, v, w, T, S)`

"""
@inline apply_z_top_bc!(top_flux::BC{<:Flux}, i, j, grid, ϕ, Gϕ, κ, t, iteration, u, v, w, T, S) =
    Gϕ[i, j, 1] -= getbc(top_flux, i, j, grid, t, iteration, u, v, w, T, S) / grid.Δz

@inline apply_z_top_bc!(top_gradient::BC{<:Gradient}, i, j, grid, ϕ, Gϕ, κ, t, iteration, u, v, w, T, S) =
    Gϕ[i, j, 1] += κ * getbc(top_gradient, i, j, grid, t, iteration, u, v, w, T, S) / grid.Δz

@inline apply_z_top_bc!(top_value::BC{<:Value}, i, j, grid, ϕ, Gϕ, κ, t, iteration, u, v, w, T, S) =
    Gϕ[i, j, 1] += 2κ / grid.Δz * (getbc(top_value, i, j, grid, t, iteration, u, v, w, T, S) - ϕ[i, j, 1])

"""
    apply_z_bottom_bc!(bottom_bc, i, j, grid, ϕ, Gϕ, κ, t, iteration, u, v, w, T, S)

Add the part of flux divergence associated with a bottom boundary condition on ϕ.
to Gϕ, where the conservation equation for ϕ is ∂ϕ/∂t = Gϕ.
If `bottom_bc.condition` is a function, the function must have the signature

    `bottom_bc.condition(i, j, grid, t, iteration, u, v, w, T, S)`

"""
@inline apply_z_bottom_bc!(bottom_flux::BC{<:Flux}, i, j, grid, ϕ, Gϕ, κ, t, iteration, u, v, w, T, S) =
    Gϕ[i, j, grid.Nz] += getbc(bottom_flux, i, j, grid, t, iteration, u, v, w, T, S) / grid.Δz

@inline apply_z_bottom_bc!(bottom_gradient::BC{<:Gradient}, i, j, grid, ϕ, Gϕ, κ, t, iteration, u, v, w, T, S) =
    Gϕ[i, j, grid.Nz] -= κ * getbc(bottom_gradient, i, j, grid, t, iteration, u, v, w, T, S) / grid.Δz

@inline apply_z_bottom_bc!(bottom_value::BC{<:Value}, i, j, grid, ϕ, Gϕ, κ, t, iteration, u, v, w, T, S) =
    Gϕ[i, j, grid.Nz] -= 2κ / grid.Δz * (ϕ[i, j, grid.Nz] - getbc(bottom_value, i, j, grid, t, iteration, u, v, w, T, S))

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

"""
    apply_z_bcs!(top_bc, bottom_bc, grid, ϕ, Gϕ, κ, closure, eos, g, t, iteration, u, v, w, T, S)

Apply a top and/or bottom boundary condition to variable ϕ. Note that this kernel
must be launched on the GPU with blocks=(Bx, By). If launched with blocks=(Bx, By, Bz),
the boundary condition will be applied Bz times!
"""
function apply_z_bcs!(top_bc, bottom_bc, grid, ϕ, Gϕ, κ, closure, eos, g, t, iteration, u, v, w, T, S)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)

               κ_top = κ(i, j, 1,       grid, closure, eos, g, u, v, w, T, S)
            κ_bottom = κ(i, j, grid.Nz, grid, closure, eos, g, u, v, w, T, S)

               apply_z_top_bc!(top_bc,    i, j, grid, ϕ, Gϕ, κ_top,    t, iteration, u, v, w, T, S)
            apply_z_bottom_bc!(bottom_bc, i, j, grid, ϕ, Gϕ, κ_bottom, t, iteration, u, v, w, T, S)

        end
    end
    @synchronize
end
