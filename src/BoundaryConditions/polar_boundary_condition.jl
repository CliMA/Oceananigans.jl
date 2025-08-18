using Oceananigans.Grids: inactive_node, new_data

struct PolarValue{D, S}
    data :: D
    side :: S
end

Adapt.adapt_structure(to, pv::PolarValue) = PolarValue(Adapt.adapt(to, pv.data), nothing)

const PolarValueBoundaryCondition{V} = BoundaryCondition{<:Value, <:PolarValue}
const PolarOpenBoundaryCondition{V}  = BoundaryCondition{<:Open,  <:PolarValue}

function PolarValueBoundaryCondition(grid, side, LZ)
    FT   = eltype(grid)
    loc  = (Nothing, Nothing, LZ)
    data = new_data(FT, grid, loc)
    return ValueBoundaryCondition(PolarValue(data, side))
end

function PolarOpenBoundaryCondition(grid, side, LZ)
    FT   = eltype(grid)
    loc  = (Nothing, Nothing, LZ)
    data = new_data(FT, grid, loc)
    return OpenBoundaryCondition(PolarValue(data, side))
end

const PolarBoundaryCondition = Union{PolarValueBoundaryCondition, PolarOpenBoundaryCondition}

maybe_polar_boundary_condition(grid, side, ::Nothing, ℓz::LZ) where LZ = nothing
maybe_polar_boundary_condition(grid, side, ::Center,  ℓz::LZ) where LZ = PolarValueBoundaryCondition(grid, side, LZ)
maybe_polar_boundary_condition(grid, side, ::Face,    ℓz::LZ) where LZ = PolarOpenBoundaryCondition(grid, side, LZ)

# Just a column
@inline getbc(pv::PolarValue, i, k, args...) = @inbounds pv.data[1, 1, k]

@kernel function _average_pole_value!(data, c, j, grid, loc)
    i′, j′, k = @index(Global, NTuple)
    c̄ = zero(grid)
    n = 0
    @inbounds for i in 1:grid.Nx
        inactive = inactive_node(i, j, k, grid, loc...)
        c̄ += ifelse(inactive, 0, c[i, j, k])
        n += ifelse(inactive, 0, 1)
    end
    @inbounds data[i′, j′, k] = ifelse(n == 0,  0,  c̄ / n)
end

function update_pole_value!(bc::PolarValue, c, grid, loc)
    j = bc.side == :north ? grid.Ny : 1
    Nz = size(c, 3)
    Oz = c.offsets[3]
    params = KernelParameters(1:1, 1:1, 1+Oz:Nz+Oz)
    launch!(architecture(bc.data), grid, params, _average_pole_value!, bc.data, c, j, grid, loc)
    return nothing
end

const SouthPolarBC = Tuple{<:PolarBoundaryCondition, <:BoundaryCondition}
const NorthPolarBC = Tuple{<:BoundaryCondition, <:PolarBoundaryCondition}
const SouthAndNorthPolarBC = Tuple{<:PolarBoundaryCondition, <:PolarBoundaryCondition}

# fill_halo_event!(c, kernels![task], bcs[task], loc, grid, args...; kwargs...)
function fill_halo_event!(c, kernel!, bc::PolarBoundaryCondition, loc, grid, args...; kwargs...)
    update_pole_value!(bc.condition, c, grid, loc)
    return kernel!(c, bc, loc, grid, Tuple(args))
end

function fill_halo_event!(c, kernel!, bcs::SouthPolarBC, loc, grid, args...; kwargs...)
    update_pole_value!(bcs[1].condition, c, grid, loc)
    return kernel!(c, bcs..., loc, grid, Tuple(args))
end

function fill_halo_event!(c, kernel!, bcs::NorthPolarBC, loc, grid, args...; kwargs...)
    update_pole_value!(bcs[2].condition, c, grid, loc)
    return kernel!(c, bcs..., loc, grid, Tuple(args))
end

function fill_halo_event!(c, kernel!, bcs::SouthAndNorthPolarBC, loc, grid, args...; kwargs...)
    update_pole_value!(bcs[1].condition, c, grid, loc)
    update_pole_value!(bcs[2].condition, c, grid, loc)
    return kernel!(c, bcs..., loc, grid, Tuple(args))
end

