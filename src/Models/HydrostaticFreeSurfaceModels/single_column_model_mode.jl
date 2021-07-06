using KernelAbstractions: NoneEvent

using Oceananigans.Grids: Flat, Bounded

import Oceananigans.Utils: launch!

#####
##### Implements a "single column model mode" for HydrostaticFreeSurfaceModel
#####

const SingleColumnGrid = AbstractGrid{<:AbstractFloat, <:Flat, <:Flat, <:Bounded}

PressureField(arch, ::SingleColumnGrid) = (pHY′ = nothing,)
FreeSurface(free_surface::ExplicitFreeSurface{Nothing}, velocities, arch, ::SingleColumnGrid) = nothing
FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, velocities, arch, ::SingleColumnGrid) = nothing

validate_momentum_advection(momentum_advection, ::SingleColumnGrid) = nothing
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, ::SingleColumnGrid) = nothing, NamedTuple()
validate_tracer_advection(tracer_advection::Nothing, ::SingleColumnGrid) = nothing, NamedTuple()

@inline hydrostatic_pressure_y_gradient(i, j, k, grid, ::Nothing) = zero(eltype(grid))

@inline launch!(arch, ::SingleColumnGrid, ::Val{:xy},  args...; kwargs...) = NoneEvent()
@inline launch!(arch, ::SingleColumnGrid, ::Val{dims}, args...; kwargs...) where dims = launch!(arch, grid, dims, args...; kwargs...)
