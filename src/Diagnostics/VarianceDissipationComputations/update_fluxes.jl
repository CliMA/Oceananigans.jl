using Oceananigans: fields
using Oceananigans.Grids: topology, Flat

# Store advective and diffusive fluxes for dissipation computation
function cache_fluxes!(model, dissipation)
    grid   = model.grid
    arch   = architecture(grid)
    sz     = size(model.tracers[1].data)
    of     = model.tracers[1].data.offsets

    params = KernelParameters(sz, of)
    
    Uⁿ   = dissipation.previous_state.Uⁿ
    Uⁿ⁻¹ = dissipation.previous_state.Uⁿ⁻¹ 
    U    = model.velocities

    launch!(arch, grid, params, _update_transport!, Uⁿ, Uⁿ⁻¹, grid, U)

    for (tracer_id, tracer_name) in enumerate(keys(dissipation.advective_production))
        cache_fluxes!(dissipation, model, tracer_name, tracer_id)
    end

    return nothing
end

function flux_parameters(grid)
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topology(grid)
    Fx = ifelse(TX == Flat, 1:1, 1:Nx+1)
    Fy = ifelse(TY == Flat, 1:1, 1:Ny+1)
    Fz = ifelse(TZ == Flat, 1:1, 1:Nz+1)
    return KernelParameters(Fx, Fy, Fz)
end

function cache_fluxes!(dissipation, model, tracer_name::Symbol, tracer_id)
    
    # Grab tracer properties
    c    = model.tracers[tracer_name]
    cⁿ⁻¹ = dissipation.previous_state[tracer_name]

    grid = model.grid
    arch = architecture(grid)
    U = model.velocities
    params = flux_parameters(grid)

    ####
    #### Update the advective fluxes and compute gradient squared
    ####

    Fⁿ   = dissipation.advective_fluxes.Fⁿ[tracer_name]
    Fⁿ⁻¹ = dissipation.advective_fluxes.Fⁿ⁻¹[tracer_name]
    Gⁿ   = dissipation.gradient_squared[tracer_name]
    advection = getadvection(model.advection, tracer_name)

    launch!(arch, grid, params, _cache_advective_fluxes!, Gⁿ, Fⁿ, Fⁿ⁻¹, grid, advection, U, c)

    ####
    #### Update the diffusive fluxes
    ####

    Vⁿ   = dissipation.diffusive_fluxes.Vⁿ[tracer_name]
    Vⁿ⁻¹ = dissipation.diffusive_fluxes.Vⁿ⁻¹[tracer_name]

    D = model.diffusivity_fields
    B = model.buoyancy
    clk  = model.clock
    clo  = model.closure
    model_fields = fields(model)

    launch!(arch, grid, params, _cache_diffusive_fluxes!, Vⁿ, Vⁿ⁻¹, grid, clo, D, B, c, tracer_id, clk, model_fields)

    parent(cⁿ⁻¹) .= parent(c)

    return nothing
end

@kernel function _update_transport!(Uⁿ, Uⁿ⁻¹, grid, U)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Uⁿ⁻¹.u[i, j, k] = Uⁿ.u[i, j, k]
        Uⁿ⁻¹.v[i, j, k] = Uⁿ.v[i, j, k]
        Uⁿ⁻¹.w[i, j, k] = Uⁿ.w[i, j, k]
          Uⁿ.u[i, j, k] = U.u[i, j, k] * Axᶠᶜᶜ(i, j, k, grid) 
          Uⁿ.v[i, j, k] = U.v[i, j, k] * Ayᶜᶠᶜ(i, j, k, grid) 
          Uⁿ.w[i, j, k] = U.w[i, j, k] * Azᶜᶜᶠ(i, j, k, grid) 
    end
end
