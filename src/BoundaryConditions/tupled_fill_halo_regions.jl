#####
##### Filling the halos of a tuple of data arrays with one kernel launch per side
#####

tupled_fill_halo_regions!(c::Tuple, kernels!, bcs::Tuple, loc, grid; kwargs...) =
    tupled_fill_halo_events!(c, values(kernels!), map(values, bcs), loc, grid; kwargs...)

@inline tupled_fill_halo_events!(c, ::Tuple{}, bcs, loc, grid; kwargs...) = nothing

@inline function tupled_fill_halo_events!(c, kernels!::Tuple, bcs, loc, grid; kwargs...)
    tupled_fill_halo_event!(c, first(kernels!), map(first, bcs), loc, grid; kwargs...)
    tupled_fill_halo_events!(c, Base.tail(kernels!), map(Base.tail, bcs), loc, grid; kwargs...)
    return nothing
end

@inline tupled_fill_halo_event!(c, ::Nothing, bcs, loc, grid; kwargs...) = nothing

@inline tupled_fill_halo_event!(c, pfh::PeriodicFillHalo{K, N, H}, bcs, loc, grid; kwargs...) where {K, N, H} =
    pfh.kernel(c, Val(N), Val(H))

@inline function tupled_fill_halo_event!(c, kernel!, bcs, loc, grid; fill_normal_flow_bcs=true, kwargs...)
    if fill_normal_flow_bcs
        launch_tupled_fill_halo!(c, kernel!, bcs, loc, grid)
    else
        launch_tupled_fill_halo!(c, kernel!, map(without_normal_flow, bcs), loc, grid)
    end
    return nothing
end

# As in `fill_halo_event!`, an array is skipped on sides whose boundary conditions are all unfilled
@inline without_normal_flow(bcs::Tuple{Any, Any}) = fills_halo(bcs[1], false) | fills_halo(bcs[2], false) ? bcs : (nothing, nothing)
@inline without_normal_flow(bcs::Tuple{Any}) = fills_halo(bcs[1], false) ? bcs : (nothing,)

@inline launch_tupled_fill_halo!(c, kernel!, bcs::Tuple{Tuple{Any, Any}, Vararg}, loc, grid) = kernel!(c, map(first, bcs), map(last, bcs), loc, grid, ())
@inline launch_tupled_fill_halo!(c, kernel!, bcs::Tuple{Tuple{Any}, Vararg}, loc, grid) = kernel!(c, map(first, bcs), loc, grid, ())

#####
##### Unrolled fills of tuples of data arrays, called by the fill kernels
#####

for fill_halo! in (:_fill_west_halo!, :_fill_east_halo!, :_fill_south_halo!, :_fill_north_halo!, :_fill_bottom_halo!, :_fill_top_halo!)
    @eval @inline function $fill_halo!(i, j, grid, c::Tuple, bcs::Tuple, loc, args...)
        ntuple(n -> (@inline; $fill_halo!(i, j, grid, c[n], bcs[n], loc, args...)), Val(length(c)))
        return nothing
    end
end

for fill_periodic_halo! in (:fill_periodic_west_and_east_halo!, :fill_periodic_south_and_north_halo!, :fill_periodic_bottom_and_top_halo!)
    @eval @inline function $fill_periodic_halo!(i, j, c::Tuple, ::Val{N}, ::Val{H}) where {N, H}
        ntuple(n -> (@inline; $fill_periodic_halo!(i, j, c[n], Val(N), Val(H))), Val(length(c)))
        return nothing
    end
end
