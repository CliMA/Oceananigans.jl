"""
Compute total density from densities of massive tracers
"""
function update_total_density!(total_density, grid, gases, tracers)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            total_density[i, j, k] = diagnose_density(i, j, k, grid, gases, tracers)
        end
    end
end

update_total_density!(model) =
    update_total_density!(model.total_density, model.grid, model.gases, model.tracers)

"""
Slow forcings include viscous dissipation, diffusion, and Coriolis terms.
"""
function compute_slow_source_terms!(slow_source_terms, grid, thermodynamic_variable, gases, gravity, coriolis, closure, total_density, momenta, tracers, diffusivities, forcing, clock)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            slow_source_terms.ρu[i, j, k] = ρu_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.u(i, j, k, grid, clock, nothing)
            slow_source_terms.ρv[i, j, k] = ρv_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.v(i, j, k, grid, clock, nothing)
            slow_source_terms.ρw[i, j, k] = ρw_slow_source_term(i, j, k, grid, coriolis, closure, total_density, momenta, diffusivities) + forcing.w(i, j, k, grid, clock, nothing)
        end

        for (tracer_index, ρc_name) in enumerate(propertynames(tracers))
            ρc   = getproperty(tracers, ρc_name)
            S_ρc = getproperty(slow_source_terms.tracers, ρc_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                S_ρc[i, j, k] = ρc_slow_source_term(i, j, k, grid, closure, tracer_index, total_density, ρc, diffusivities)
            end
        end

        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            slow_source_terms.tracers[1].data[i, j, k] += ρt_slow_source_term(i, j, k, grid, closure, thermodynamic_variable, gases, gravity, total_density, momenta, tracers, diffusivities)
        end

    end
end

"""
Fast forcings include advection, pressure gradient, and buoyancy terms.
"""
function compute_fast_source_terms!(fast_source_terms, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            fast_source_terms.ρu[i, j, k] = ρu_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρu)
            fast_source_terms.ρv[i, j, k] = ρv_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρv)
            fast_source_terms.ρw[i, j, k] = ρw_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, advection_scheme, total_density, momenta, tracers, slow_source_terms.ρw)
        end

        for ρc_name in propertynames(tracers)
            ρc   = getproperty(tracers, ρc_name)
            F_ρc = getproperty(fast_source_terms.tracers, ρc_name)
            S_ρc = getproperty(slow_source_terms.tracers, ρc_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                F_ρc[i, j, k] = ρc_fast_source_term(i, j, k, grid, advection_scheme, total_density, momenta, ρc, S_ρc)
            end
        end

        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            fast_source_terms.tracers[1].data[i, j, k] += ρt_fast_source_term(i, j, k, grid, thermodynamic_variable, gases, gravity, total_density, momenta, tracers)
        end
    end
end

"""
Updates variables according to the RK3 time step:
    1. Φ*      = Φᵗ + Δt/3 * R(Φᵗ)
    2. Φ**     = Φᵗ + Δt/2 * R(Φ*)
    3. Φ(t+Δt) = Φᵗ + Δt   * R(Φ**)
"""
function advance_variables!(state_variables, grid, momenta, tracers, fast_source_terms; Δt)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            state_variables.ρu[i, j, k] = momenta.ρu[i, j, k] + Δt * fast_source_terms.ρu[i, j, k]
            state_variables.ρv[i, j, k] = momenta.ρv[i, j, k] + Δt * fast_source_terms.ρv[i, j, k]
            state_variables.ρw[i, j, k] = momenta.ρw[i, j, k] + Δt * fast_source_terms.ρw[i, j, k]
        end

        for ρc_name in propertynames(tracers)
            ρc  = getproperty(tracers, ρc_name)
            I_ρc = getproperty(state_variables.tracers, ρc_name)
            F_ρc = getproperty(fast_source_terms.tracers, ρc_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                I_ρc[i, j, k] = ρc[i, j, k] + Δt * F_ρc[i, j, k]
            end
        end
    end
end
