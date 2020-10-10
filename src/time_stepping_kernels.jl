"""
    update_total_density!(total_density, grid, gases, tracers)

Compute total density from densities of massive tracers.
"""
function update_total_density!(total_density, grid, gases, tracers)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds total_density[i, j, k] = diagnose_density(i, j, k, grid, gases, tracers)
    end
    return nothing
end

update_total_density!(model) =
    update_total_density!(model.total_density, model.grid, model.gases, model.tracers)

#####
##### Computing slow source terms (viscous dissipation, diffusion, and Coriolis terms).
#####

function compute_slow_source_terms!(slow_source_terms, grid, thermodynamic_variable, gases, gravity, coriolis, closure, total_density, momenta, tracers, diffusivities, forcing, clock)
    
    compute_slow_momentum_source_terms!(slow_source_terms, grid, coriolis, closure, total_density, momenta, diffusivities, forcing, clock)
    
    for (tracer_index, ρc_name) in enumerate(propertynames(tracers))
        ρc   = getproperty(tracers, ρc_name)
        S_ρc = getproperty(slow_source_terms.tracers, ρc_name)
        compute_slow_tracer_source_terms!(S_ρc, grid, closure, tracer_index, total_density, ρc, diffusivities, forcing, clock)
    end
    
    compute_slow_thermodynamic_variable_source_terms!(slow_source_terms.tracers[1], grid, thermodynamic_variable, gases, gravity, closure, total_density, momenta, tracers, diffusivities)
    
    return nothing
end

function compute_slow_momentum_source_terms!(slow_source_terms, grid, coriolis, closure, total_density, momenta, diffusivities, forcing, clock)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds slow_source_terms.ρu[i, j, k] = ρu_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.u(i, j, k, grid, clock, nothing)
        @inbounds slow_source_terms.ρv[i, j, k] = ρv_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.v(i, j, k, grid, clock, nothing)
        @inbounds slow_source_terms.ρw[i, j, k] = ρw_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.w(i, j, k, grid, clock, nothing)
    end
    return nothing
end

function compute_slow_tracer_source_terms!(S_ρc, grid, closure, tracer_index, total_density, ρc, diffusivities, forcing, clock)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds S_ρc[i, j, k] = ρc_slow_source_term(i, j, k, grid, closure, tracer_index, total_density, ρc, diffusivities) # + forcing_ρc(i, j, k, grid, clock, nothing)
    end
    return nothing
end

function compute_slow_thermodynamic_variable_source_terms!(S_ρt, grid, thermodynamic_variable, gases, gravity, closure, total_density, momenta, tracers, diffusivities)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds S_ρt.data[i, j, k] += ρt_slow_source_term(i, j, k, grid, closure, thermodynamic_variable, gases, gravity, total_density, momenta, tracers, diffusivities)
    end
    return nothing
end

#####
##### Computing fast source terms (advection, pressure gradient, and buoyancy terms).
#####


function compute_fast_source_terms!(fast_source_terms, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms)

    compute_fast_momentum_source_terms!(fast_source_terms, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms)

    for ρc_name in propertynames(tracers)
        ρc   = getproperty(tracers, ρc_name)
        F_ρc = getproperty(fast_source_terms.tracers, ρc_name)
        S_ρc = getproperty(slow_source_terms.tracers, ρc_name)
        compute_fast_tracer_source_terms!(F_ρc, grid, advection_scheme, total_density, momenta, ρc, S_ρc)
    end
    
    compute_fast_thermodynamic_source_terms!(fast_source_terms.tracers[1], grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)

    return nothing
end

function compute_fast_momentum_source_terms!(fast_source_terms, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds fast_source_terms.ρu[i, j, k] = ρu_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρu)
        @inbounds fast_source_terms.ρv[i, j, k] = ρv_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρv)
        @inbounds fast_source_terms.ρw[i, j, k] = ρw_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρw)
    end
    return nothing
end

function compute_fast_tracer_source_terms!(F_ρc, grid, advection_scheme, total_density, momenta, ρc, S_ρc)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds F_ρc[i, j, k] = ρc_fast_source_term(i, j, k, grid, advection_scheme, total_density, momenta, ρc, S_ρc)
    end
    return nothing
end

function compute_fast_thermodynamic_source_terms!(F_ρt, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds F_ρt[i, j, k] += ρt_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
    end
    return nothing
end

#####
##### Advancing state variables
#####

function advance_state_variables!(state_variables, grid, momenta, tracers, fast_source_terms; Δt)
    
    advance_momentum!(state_variables, grid, momenta, fast_source_terms, Δt)
    
    for ρc_name in propertynames(tracers)
        ρc   = getproperty(tracers, ρc_name)
        ρc⁺  = getproperty(state_variables.tracers, ρc_name)
        F_ρc = getproperty(fast_source_terms.tracers, ρc_name)
        advance_tracer!(ρc⁺, grid, ρc, F_ρc, Δt)
    end

    return nothing
end

function advance_momentum!(momenta⁺, grid, momenta, fast_source_terms, Δt)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds momenta⁺.ρu[i, j, k] = momenta.ρu[i, j, k] + Δt * fast_source_terms.ρu[i, j, k]
        @inbounds momenta⁺.ρv[i, j, k] = momenta.ρv[i, j, k] + Δt * fast_source_terms.ρv[i, j, k]
        @inbounds momenta⁺.ρw[i, j, k] = momenta.ρw[i, j, k] + Δt * fast_source_terms.ρw[i, j, k]
    end
    return nothing
end

function advance_tracer!(ρc⁺, grid, ρc, F_ρc, Δt)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds ρc⁺[i, j, k] = ρc[i, j, k] + Δt * F_ρc[i, j, k]
    end
    return nothing
end
