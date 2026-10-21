#####
##### Filling the halos of several data arrays with one kernel launch per side
#####

"""
$(TYPEDSIGNATURES)

Fill the halo regions of the tuple of data arrays `c`, whose boundary conditions are the tuple
`bcs` of ordered boundary conditions (`FieldBoundaryConditions.ordered_bcs`). The arrays share
the location `loc`, the `grid`, and preconfigured fill `kernels!` of the same type, so every side
is filled for all arrays with a single launch of the kernel that fills one array: the work for each
array is unrolled within the kernel, in the order of `c`. The sides are filled in the order of
`kernels!`, as for a single array.

The boundary conditions must not read the trailing arguments of `fill_halo_regions!` (see
`needs_halo_fill_args`), nor require work before the kernels are launched.
"""
function fill_batched_halo_regions!(c::Tuple, kernels!, bcs::Tuple, loc, grid; kwargs...)
    side_ordered_bcs = map(values, bcs)
    batched_fill_halo_events!(c, values(kernels!), side_ordered_bcs, loc, grid; kwargs...)
    return nothing
end

@inline batched_fill_halo_events!(c, ::Tuple{}, bcs, loc, grid; kwargs...) = nothing

@inline function batched_fill_halo_events!(c, kernels!::Tuple, bcs, loc, grid; kwargs...)
    batched_fill_halo_event!(c, first(kernels!), map(first, bcs), loc, grid; kwargs...)
    batched_fill_halo_events!(c, Base.tail(kernels!), map(Base.tail, bcs), loc, grid; kwargs...)
    return nothing
end

@inline batched_fill_halo_event!(c, ::Nothing, bcs, loc, grid; kwargs...) = nothing

@inline batched_fill_halo_event!(c, pfh::PeriodicFillHalo{K, N, H}, bcs, loc, grid; kwargs...) where {K, N, H} =
    pfh.kernel(c, Val(N), Val(H))

@inline function batched_fill_halo_event!(c, kernel!, bcs, loc, grid; fill_normal_flow_bcs=true, kwargs...)
    if fill_normal_flow_bcs
        launch_batched_fill_halo!(c, kernel!, bcs, loc, grid)
    else
        launch_batched_fill_halo!(c, kernel!, map(without_normal_flow, bcs), loc, grid)
    end
    return nothing
end

# Mirror `fill_halo_event!`: an array is skipped on sides whose boundary conditions are all skipped
@inline without_normal_flow(bcs::Tuple{Any, Any}) = fills_halo(bcs[1], false) | fills_halo(bcs[2], false) ? bcs : (nothing, nothing)
@inline without_normal_flow(bcs::Tuple{Any}) = fills_halo(bcs[1], false) ? bcs : (nothing,)

const UnfilledTwoSidedBCs = Tuple{Tuple{Nothing, Nothing}, Vararg{Tuple{Nothing, Nothing}}}
const UnfilledOneSidedBCs = Tuple{Tuple{Nothing}, Vararg{Tuple{Nothing}}}
const TwoSidedBCs = Tuple{Tuple{Any, Any}, Vararg{Tuple{Any, Any}}}
const OneSidedBCs = Tuple{Tuple{Any}, Vararg{Tuple{Any}}}

@inline launch_batched_fill_halo!(c, kernel!, ::UnfilledTwoSidedBCs, loc, grid) = nothing
@inline launch_batched_fill_halo!(c, kernel!, ::UnfilledOneSidedBCs, loc, grid) = nothing
@inline launch_batched_fill_halo!(c, kernel!, bcs::TwoSidedBCs, loc, grid) = kernel!(c, map(first, bcs), map(last, bcs), loc, grid, ())
@inline launch_batched_fill_halo!(c, kernel!, bcs::OneSidedBCs, loc, grid) = kernel!(c, map(first, bcs), loc, grid, ())

#####
##### Unrolled fills of tuples of data arrays, called by the fill kernels
#####

for fill_halo! in (:_fill_west_halo!, :_fill_east_halo!, :_fill_south_halo!, :_fill_north_halo!, :_fill_bottom_halo!, :_fill_top_halo!)
    @eval @inline function $fill_halo!(i, j, grid, c::Tuple, bcs::Tuple, loc, args...)
        ntuple(Val(length(c))) do n
            Base.@_inline_meta
            @inbounds $fill_halo!(i, j, grid, c[n], bcs[n], loc, args...)
            return nothing
        end
        return nothing
    end
end

for fill_periodic_halo! in (:fill_periodic_west_and_east_halo!, :fill_periodic_south_and_north_halo!, :fill_periodic_bottom_and_top_halo!)
    @eval @inline function $fill_periodic_halo!(i, j, c::Tuple, N::Val, H::Val)
        ntuple(Val(length(c))) do n
            Base.@_inline_meta
            @inbounds $fill_periodic_halo!(i, j, c[n], N, H)
            return nothing
        end
        return nothing
    end
end
