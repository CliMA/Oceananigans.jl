abstract type MatchingScheme end

# generic infrastructure
const MOBC = BoundaryCondition{Open{<:MatchingScheme}}

# nudging

struct Nudging{IT, OT} <: MatchingScheme
     inflow_nudging_timescale :: IT
    outflow_nudging_timescale :: OT
end

(nudging::Nudging)() = nudging

const NOBC = BoundaryCondition{<:Open{<:Nudging}}

@inline function _fill_west_halo!(j, k, grid, c, bc::NOBC, loc, clock, model_fields)
    Δt = clock.last_Δt

    matching_scheme = bc.classification.matching_scheme

    i, i′ = domain_boundary_indices(LeftBoundary(), grid.Nx)

    external_state = getbc(bc, j, k, grid, clock, model_fields)
    
    wall_normal_velocity = @inbounds model_fields.u[i, j, k]

    relaxation_timescale = wall_normal_velocity > 0 * matching_scheme.outflow_nudging_timescale + 
                           wall_normal_velocity < 0 * matching_scheme.inflow_nudging_timescale

    relaxation_rate = min(1, Δt / relaxation_timescale)

    internal_state = @inbounds c[i, j, k]

    @inbounds c[i′, j, k] = 0#internal_state * (1 - relaxation_rate) + external_state * relaxation_rate
end

@inline function _fill_east_halo!(j, k, grid, c, bc::NOBC, loc, clock, model_fields)
    Δt = clock.last_Δt

    matching_scheme = bc.classification.matching_scheme

    i, i′ = domain_boundary_indices(RightBoundary(), grid.Nx)

    external_state = getbc(bc, j, k, grid, clock, model_fields)
    
    wall_normal_velocity = - @inbounds model_fields.u[i + 1, j, k]

    relaxation_timescale = wall_normal_velocity > 0 * matching_scheme.outflow_nudging_timescale + 
                           wall_normal_velocity < 0 * matching_scheme.inflow_nudging_timescale

    relaxation_rate = min(1, Δt / relaxation_timescale)

    internal_state = @inbounds c[i + 1, j, k]

    @inbounds c[i′ + 1, j, k] = 0#internal_state * (1 - relaxation_rate) + external_state * relaxation_rate
end

#=
@inline  _fill_south_halo!(i, k, grid, c, bc::MOBC, loc, args...) = @inbounds c[i, 1, k]           = getbc(bc, i, k, grid, args...)
@inline  _fill_north_halo!(i, k, grid, c, bc::MOBC, loc, args...) = @inbounds c[i, grid.Ny + 1, k] = getbc(bc, i, k, grid, args...)
@inline _fill_bottom_halo!(i, j, grid, c, bc::MOBC, loc, args...) = @inbounds c[i, j, 1]           = getbc(bc, i, j, grid, args...)
@inline    _fill_top_halo!(i, j, grid, c, bc::MOBC, loc, args...) = @inbounds c[i, j, grid.Nz + 1] = getbc(bc, i, j, grid, args...)
=#