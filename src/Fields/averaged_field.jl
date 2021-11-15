using Adapt
using Statistics
using Oceananigans.Grids
using Oceananigans.Grids: interior_parent_indices

struct AveragedField{X, Y, Z, S, A, D, G, T, N, O} <: AbstractReducedField{X, Y, Z, A, G, T, N}
            data :: D
    architecture :: A
            grid :: G
            dims :: NTuple{N, Int}
         operand :: O
          status :: S

    function AveragedField{X, Y, Z}(data::D, arch::A, grid::G, dims, operand::O;
                                    recompute_safely=true) where {X, Y, Z, D, A, G, O}

        dims = validate_reduced_dims(dims)
        validate_reduced_locations(X, Y, Z, dims)
        validate_field_data(X, Y, Z, data, grid)

        status = recompute_safely ? nothing : FieldStatus(0.0)

        S = typeof(status)
        N = length(dims)
        T = eltype(grid)

        return new{X, Y, Z, S, A, D, G, T, N, O}(data, arch, grid, dims, operand, status)
    end

    function AveragedField{X, Y, Z}(data::D, arch::A, grid::G, dims, operand::O, status::S) where {X, Y, Z, D, A, G, O, S}
        return new{X, Y, Z, S, A, D, G, eltype(grid), length(dims), O}(data, arch, grid, dims, operand, status)
    end
end

"""
    AveragedField(operand::AbstractField; dims, data=nothing, recompute_safely=false)

Returns an AveragedField averaged over `dims`. `dims` is a tuple of integers indicating
spatial dimensions; in a Cartesian coordinate system, `1=x, `2=y`, and `3=z`.

Arguments
=========

- `dims`: Tuple of integers specifying the dimensions to average `operand`.
          A single integer is also accepted for averaging over a single dimension.  

- `data`: An `OffsetArray` for storing averaged data.
          Useful if carefully managing memory allocation.
          If unspecified, `data` is created by `Oceananigans.Grids.new_data`.

- `recompute_safely`: A boolean that's relevant only if the `AveragedField` is used
                      within another computation. If `recompute_safely=false`,
                      `AveragedField` will *not* be recomputed before computing any dependent
                      computations if `AveragedField.status` is consistent with the current state of the simulation.
                      If `recompute_safely=true`, `AveragedField` is always recomputed
                      before performing a dependent computation.

Examples
=======

```julia
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 2, 2), x=(0, 1), y=(0, 1), z=(0, 1));

julia> c = CenterField(CPU(), grid);

julia> C_xy = AveragedField(c, dims=(1, 2)) # average over x, y
AveragedField over dims=(1, 2) located at (⋅, ⋅, Center) of Field located at (Center, Center, Center)
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (1, 1, 2)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=2, Ny=2, Nz=2)
├── dims: (1, 2)
├── operand: Field located at (Center, Center, Center)
└── status: time=0.0

julia> C_z = AveragedField(c, dims=3) # averaged over z
AveragedField over dims=(3,) located at (Center, Center, ⋅) of Field located at (Center, Center, Center)
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (2, 2, 1)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=2, Ny=2, Nz=2)
├── dims: (3,)
├── operand: Field located at (Center, Center, Center)
└── status: time=0.0
```
"""
function AveragedField(operand::AbstractField; dims, data=nothing, recompute_safely=true)

    arch = architecture(operand)
    loc = reduced_location(location(operand), dims=dims)
    grid = operand.grid

    if isnothing(data)
        data = new_data(arch, grid, loc)
        recompute_safely = false
    end

    return AveragedField{loc[1], loc[2], loc[3]}(data, arch, grid, dims, operand,
                                                 recompute_safely=recompute_safely)
end

"""
    compute!(avg::AveragedField, time=nothing)

Compute the average of `avg.operand` and store the result in `avg.data`.
"""
function compute!(avg::AveragedField, time=nothing)
    compute_at!(avg.operand, time)
    mean!(avg, avg.operand)
    return nothing
end

compute_at!(avg::AveragedField{X, Y, Z, <:FieldStatus}, time) where {X, Y, Z} =
    conditional_compute!(avg, time)

#####
##### Adapt
#####

Adapt.adapt_structure(to, averaged_field::AveragedField{X, Y, Z}) where {X, Y, Z} =
    AveragedField{X, Y, Z}(Adapt.adapt(to, averaged_field.data), nothing,
                           nothing, averaged_field.dims, nothing, nothing)
