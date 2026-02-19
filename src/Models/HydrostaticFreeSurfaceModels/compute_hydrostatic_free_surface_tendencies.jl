import Oceananigans: tracer_tendency_kernel_function
import Oceananigans.Models: interior_tendency_kernel_parameters

using Oceananigans: fields, prognostic_fields, TendencyCallsite, UpdateStateCallsite
using Oceananigans.Grids: halo_size
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Biogeochemistry: update_tendencies!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE, FlavorOfTD

using Oceananigans.Grids: get_active_cells_map, active_cell
using Oceananigans.Architectures: CPU
import Oceananigans.Architectures as AC
using Oceananigans.Fields: Field, interior
using Oceananigans.Advection: StaticWENO, StaticWENOVectorInvariant
using KernelAbstractions: @kernel, @index

"""
    compute_momentum_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

Compute tendencies for horizontal velocity fields `u` and `v`.

This function:
1. Computes interior momentum tendencies (advection, Coriolis, pressure gradient, diffusion, forcing)
2. Completes halo communication and computes buffer tendencies for distributed grids
3. Computes flux boundary condition contributions
4. Executes any callbacks with `TendencyCallsite`

Momentum tendencies are stored in `model.timestepper.Gⁿ.u` and `model.timestepper.Gⁿ.v`.
"""
function compute_momentum_tendencies!(model::HydrostaticFreeSurfaceModel, callbacks)

    grid = model.grid
    arch = architecture(grid)

    active_cells_map = get_active_cells_map(model.grid, Val(:interior))
		
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters; active_cells_map)
    complete_communication_and_compute_momentum_buffer!(model, grid, arch)

    for callback in callbacks
        callback.callsite isa TendencyCallsite && callback(model)
    end

    return nothing
end

"""
    compute_tracer_tendencies!(model::HydrostaticFreeSurfaceModel)

Compute tendencies for all tracer fields.

This function:
1. Computes interior tracer tendencies (advection, diffusion, forcing, biogeochemistry sources)
2. Completes halo communication and computes buffer tendencies for distributed grids
3. Computes flux boundary condition contributions
4. Scales tendencies by the grid stretching factor for z-star coordinates
5. Updates biogeochemistry tendencies

Tracers are advected using `model.transport_velocities` which may differ from `model.velocities`
when using split-explicit free surfaces (transport velocities include barotropic correction).

Tracer tendencies are stored in `model.timestepper.Gⁿ[tracer_name]`.
"""
function compute_tracer_tendencies!(model::HydrostaticFreeSurfaceModel)

    grid = model.grid
    arch = architecture(grid)

    active_cells_map  = get_active_cells_map(model.grid, Val(:interior))
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map)
    complete_communication_and_compute_tracer_buffer!(model, grid, arch)
    compute_tracer_flux_bcs!(model)

    scale_by_stretching_factor!(model.timestepper.Gⁿ, model.tracers, model.grid)

    update_tendencies!(model.biogeochemistry, model)

    return nothing
end

# Fallback
compute_free_surface_tendency!(grid, model, free_surface) = nothing

@inline function top_tracer_boundary_conditions(grid, tracers)
    names = propertynames(tracers)
    values = Tuple(tracers[c].boundary_conditions.top for c in names)

    # Some shenanigans for type stability?
    return NamedTuple{tuple(names...)}(tuple(values...))
end

"""
    compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map=nothing)

Compute tracer tendencies in the grid interior (or on specified active cells).

Launches the tracer tendency kernel for each tracer, computing advection, diffusion,
and forcing contributions. Uses `model.transport_velocities` for advection.
"""
function compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map=nothing)

    arch = model.architecture
    grid = model.grid

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))

        @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection   = model.advection[tracer_name]
        @inbounds c_forcing     = model.forcing[tracer_name]
        @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

        args = tuple(Val(tracer_index),
                     Val(tracer_name),
                     c_advection,
                     model.closure,
                     c_immersed_bc,
                     model.buoyancy,
                     model.biogeochemistry,
                     model.transport_velocities,
                     model.free_surface,
                     model.tracers,
                     model.closure_fields,
                     model.auxiliary_fields,
                     model.clock,
                     c_forcing)

        launch!(arch, grid, kernel_parameters,
                compute_hydrostatic_free_surface_Gc!,
                c_tendency,
                grid,
                args;
                active_cells_map)
    end

    return nothing
end

function get_advection_conditioned_map(scheme, grid::ImmersedBoundaryGrid; active_cells_map=nothing)
    # Field is true if the max scheme can be used for computing u advection
    max_scheme_field = Field{Center, Center, Center}(grid, Bool)
    fill!(max_scheme_field, false)
    launch!(architecture(grid), grid, :xyz, condition_map!, max_scheme_field, grid, scheme; active_cells_map)

    return NamedTuple{(:interior, :boundary)}(split_indices(max_scheme_field, grid; active_cells_map))
end

function split_indices(field, grid; active_cells_map=nothing)
    if isnothing(active_cells_map)
        return split_indices_full(field, grid)
    else
        return split_indices_mapped(field, grid, active_cells_map)
    end
end

function split_indices_mapped(field, grid, active_cells_map)
    IndexType = Tuple{Int64,Int64,Int64}
    vals = AC.on_architecture(CPU(), interior(field))
    active_indices = AC.on_architecture(CPU(), active_cells_map)
    map1 = Vector{eltype(active_indices)}()
    map2 = Vector{eltype(active_indices)}()
    for index in active_indices
        val = vals[convert(IndexType, index)...]
	if val
	    push!(map1, index)
	else
	    push!(map2, index)
	end
    end
    println(size(map1))
    println(size(map2))
    map1 = AC.on_architecture(architecture(grid), map1) 
    map2 = AC.on_architecture(architecture(grid), map2) 
    return (map1, map2)
end

function split_indices_full(field, grid)
    IndicesType = Tuple{Int16, Int16, Int16}
    map1 = IndicesType[]
    map2 = IndicesType[]
    for k in 1:size(grid, 3)
        vals = AC.on_architecture(CPU(), interior(field, :, :, k)) 
	map1 = vcat(map1, convert_interior_indices(findall(x->x, vals), k, IndicesType))
	map2 = vcat(map2, convert_interior_indices(findall(x->!x, vals), k, IndicesType))
	GC.gc()
    end
    map1 = AC.on_architecture(architecture(grid), map1) 
    map2 = AC.on_architecture(architecture(grid), map2) 
    return (map1, map2)
end


function convert_interior_indices(interior_indices, k, IndicesType)
    interior_indices =   getproperty.(interior_indices, :I)
    interior_indices = add_3rd_index.(interior_indices, k) |> Array{IndicesType}
    return interior_indices
end

add_3rd_index(ij, k) = (ij[1], ij[2], k)

check_interior_xyz(i, j, k, ibg, scheme) = (&&)(check_interior_x(i, j, k, ibg, scheme),
                                                check_interior_y(i, j, k, ibg, scheme),
                                                check_interior_z(i, j, k, ibg, scheme))

function check_interior_x(i, j, k, ibg, scheme::AbstractAdvectionScheme{N}) where N
    interior = true
    
    buffer = N + 1
    for di in -buffer:buffer
        interior &= active_cell(i + di, j, k, ibg)
    end
    return interior
end

function check_interior_y(i, j, k, ibg, scheme::AbstractAdvectionScheme{N}) where N
    interior = true
    
    buffer = N + 1
    for dj in -buffer:buffer
        interior &= active_cell(i, j + dj, k, ibg)
    end
    return interior
end

function check_interior_z(i, j, k, ibg, scheme::AbstractAdvectionScheme{N}) where N
    interior = true
    
    buffer = N + 1
    for dk in -buffer:buffer
        interior &= active_cell(i, j, k + dk, ibg)
    end
    return interior
end

@kernel function condition_map!(max_scheme_field, ibg, scheme)
    i, j, k = @index(Global, NTuple)

    @inbounds max_scheme_field[i, j, k] = check_interior_xyz(i, j, k, ibg, scheme)
end

"""
    compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; active_cells_map=nothing)

Compute momentum tendencies for `u` and `v` in the grid interior (or on specified active cells).
"""
function compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; active_cells_map=nothing)

    grid = model.grid
    arch = architecture(grid)

		momentum_conditioned_maps = get_advection_conditioned_map(model.advection.momentum, grid; active_cells_map)
    u_conditioned_maps = momentum_conditioned_maps
		v_conditioned_maps = momentum_conditioned_maps

    u_immersed_bc = immersed_boundary_condition(velocities.u)
    v_immersed_bc = immersed_boundary_condition(velocities.v)

    u_forcing = model.forcing.u
    v_forcing = model.forcing.v

    end_momentum_kernel_args = (velocities,
                                model.free_surface,
                                model.tracers,
                                model.buoyancy,
                                model.closure_fields,
                                model.pressure.pHY′,
                                model.auxiliary_fields,
                                model.vertical_coordinate,
                                model.clock)

    static_momentum_advection = StaticWENOVectorInvariant(model.advection.momentum)

    u_kernel_args = tuple(model.coriolis, model.closure, u_immersed_bc, end_momentum_kernel_args..., u_forcing)
    u_kernel_args_tuple = NamedTuple{(:interior, :boundary)}(tuple(static_momentum_advection, u_kernel_args...), tuple(model.advection.momentum, u_kernel_args...))

    v_kernel_args = tuple(model.coriolis, model.closue, v_immersed_bc, end_momentum_kernel_args..., v_forcing)
    v_kernel_args_tuple = NamedTuple{(:interior, :boundary)}(tuple(static_momentum_advection, v_kernel_args...), tuple(model.advection.momentum, v_kernel_args...))

    launch_conditioned!(arch, grid, kernel_parameters, u_conditioned_maps,
                        compute_hydrostatic_free_surface_Gu!, (model.timestepper.Gⁿ.u, grid),
                        u_kernel_args_tuple)

    launch_conditioned!(arch, grid, kernel_parameters, v_conditioned_maps,
                        compute_hydrostatic_free_surface_Gv!, (model.timestepper.Gⁿ.v, grid),
                        v_kernel_args_tuple)

    return nothing
end

#####
##### Tendency calculators for u, v
#####

""" Calculate the right-hand-side of the u-velocity equation. """
@kernel function compute_hydrostatic_free_surface_Gu!(Gu, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid, args...)
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function compute_hydrostatic_free_surface_Gv!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

#####
##### Tendency calculators for tracers
#####

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function compute_hydrostatic_free_surface_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

