@hascuda using CUDAnative, CuArrays

import GPUifyLoops: @launch, @loop, @synchronize

using Oceananigans.Operators

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

"""
    time_step!(model, Nt, Δt)

Step forward `model` `Nt` time steps using a second-order Adams-Bashforth
method with step size `Δt`.
"""
function time_step!(model::Model{A}, Nt, Δt) where A <: Architecture
    clock = model.clock
    model_start_time = clock.time
    model_end_time = model_start_time + Nt*Δt

    if clock.iteration == 0
        for output_writer in model.output_writers
            write_output(model, output_writer)
        end
        for diagnostic in model.diagnostics
            run_diagnostic(model, diagnostic)
        end
    end

    arch = A()

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    Lx, Ly, Lz = model.grid.Lx, model.grid.Ly, model.grid.Lz
    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz

    grid = model.grid
    cfg = model.configuration
    bcs = model.boundary_conditions
    clock = model.clock

    G = model.G
    Gp = model.Gp
    constants = model.constants
    eos =  model.eos
    U = model.velocities
    tr = model.tracers
    pr = model.pressures
    forcing = model.forcing
    poisson_solver = model.poisson_solver

    δρ = model.stepper_tmp.fC1
    RHS = model.stepper_tmp.fCC1
    ϕ = model.stepper_tmp.fCC2

    gΔz = model.constants.g * model.grid.Δz
    fCor = model.constants.f

    uvw = U.u.data, U.v.data, U.w.data
    TS = tr.T.data, tr.S.data
    Guvw = G.Gu.data, G.Gv.data, G.Gw.data

    # Source terms at current (Gⁿ) and previous (G⁻) time steps.
    Gⁿ = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data
    G⁻ = Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    tb = (threads=(Tx, Ty), blocks=(Bx, By, Bz))

    for n in 1:Nt
        # Adams-Bashforth (AB2) parameter.
        χ = ifelse(model.clock.iteration == 0, -0.5, 0.125)

        # time_step_kernels!(arch(), model, χ, Δt)

        ###
        @launch device(arch) store_previous_source_terms!(grid, Gⁿ..., G⁻..., threads=(Tx, Ty), blocks=(Bx, By, Bz))
        @launch device(arch) update_buoyancy!(grid, constants, eos, δρ.data, tr.T.data, pr.pHY′.data, threads=(Tx, Ty), blocks=(Bx, By, Bz))
        @launch device(arch) calculate_interior_source_terms!(grid, constants, eos, cfg, uvw..., TS..., pr.pHY′.data, Gⁿ..., forcing, threads=(Tx, Ty), blocks=(Bx, By, Bz))
                             calculate_boundary_source_terms!(model)
        @launch device(arch) adams_bashforth_update_source_terms!(grid, Gⁿ..., G⁻..., χ, threads=(Tx, Ty), blocks=(Bx, By, Bz))
        @launch device(arch) calculate_source_term_divergence!(arch, grid, Guvw..., RHS.data, threads=(Tx, Ty), blocks=(Bx, By, Bz))

        if arch == CPU()
            solve_poisson_3d_ppn_planned!(poisson_solver, grid, RHS, ϕ)
            @. pr.pNHS.data = real(ϕ.data)
        elseif arch == GPU()
            solve_poisson_3d_ppn_gpu_planned!(Tx, Ty, Bx, By, Bz, poisson_solver, grid, RHS, ϕ)
            @launch device(arch) idct_permute!(grid, ϕ.data, pr.pNHS.data, threads=(Tx, Ty), blocks=(Bx, By, Bz))
        end

        @launch device(arch) update_velocities_and_tracers!(grid, uvw..., TS..., pr.pNHS.data, Gⁿ..., G⁻..., Δt, threads=(Tx, Ty), blocks=(Bx, By, Bz))
        ###

        clock.time += Δt
        clock.iteration += 1

        for diagnostic in model.diagnostics
            (clock.iteration % diagnostic.diagnostic_frequency) == 0 && run_diagnostic(model, diagnostic)
        end

        for output_writer in model.output_writers
            (clock.iteration % output_writer.output_frequency) == 0 && write_output(model, output_writer)
        end
    end

    return nothing
end

time_step!(model; Nt, Δt) = time_step!(model, Nt, Δt)

"Execute one time-step on the CPU."
function time_step_kernels!(arch::CPU, model, χ, Δt)
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    Lx, Ly, Lz = model.grid.Lx, model.grid.Ly, model.grid.Lz
    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz

    grid = model.grid
    cfg = model.configuration
    bcs = model.boundary_conditions
    clock = model.clock

    G = model.G
    Gp = model.Gp
    constants = model.constants
    eos =  model.eos
    U = model.velocities
    tr = model.tracers
    pr = model.pressures
    forcing = model.forcing
    poisson_solver = model.poisson_solver

    δρ = model.stepper_tmp.fC1
    RHS = model.stepper_tmp.fCC1
    ϕ = model.stepper_tmp.fCC2

    gΔz = model.constants.g * model.grid.Δz
    fCor = model.constants.f

    uvw = U.u.data, U.v.data, U.w.data
    TS = tr.T.data, tr.S.data
    Guvw = G.Gu.data, G.Gv.data, G.Gw.data

    # Source terms at current (Gⁿ) and previous (G⁻) time steps.
    Gⁿ = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data
    G⁻ = Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    store_previous_source_terms!(device(arch), grid, Gⁿ..., G⁻...)

    update_buoyancy!(device(arch), grid, constants, eos, δρ.data, tr.T.data, pr.pHY′.data)

    calculate_interior_source_terms!(device(arch), grid, constants, eos, cfg, uvw..., TS..., pr.pHY′.data, Gⁿ..., forcing)

    calculate_boundary_source_terms!(device(arch), model)

    adams_bashforth_update_source_terms!(device(arch), grid, Gⁿ..., G⁻..., χ)

    calculate_source_term_divergence!(device(arch), grid, Guvw..., RHS.data)

    solve_poisson_3d_ppn_planned!(poisson_solver, grid, RHS, ϕ)
    @. pr.pNHS.data = real(ϕ.data)

    update_velocities_and_tracers!(device(arch), grid, uvw..., TS..., pr.pNHS.data, Gⁿ..., G⁻..., Δt)

    return nothing
end

"Execute one time-step on the GPU."
function time_step_kernels!(arch::GPU, model, χ, Δt)
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    Lx, Ly, Lz = model.grid.Lx, model.grid.Ly, model.grid.Lz
    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz

    grid = model.grid
    cfg = model.configuration
    bcs = model.boundary_conditions
    clock = model.clock

    G = model.G
    Gp = model.Gp
    constants = model.constants
    eos =  model.eos
    U = model.velocities
    tr = model.tracers
    pr = model.pressures
    forcing = model.forcing
    poisson_solver = model.poisson_solver

    δρ = model.stepper_tmp.fC1
    RHS = model.stepper_tmp.fCC1
    ϕ = model.stepper_tmp.fCC2

    gΔz = model.constants.g * model.grid.Δz
    fCor = model.constants.f

    uvw = U.u.data, U.v.data, U.w.data
    TS = tr.T.data, tr.S.data
    Guvw = G.Gu.data, G.Gv.data, G.Gw.data

    # Source terms at current (Gⁿ) and previous (G⁻) time steps.
    Gⁿ = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data
    G⁻ = Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) store_previous_source_terms!(device(arch), grid, Gⁿ..., G⁻...)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) update_buoyancy!(device(arch), grid, constants, eos, δρ.data, tr.T.data, pr.pHY′.data)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_interior_source_terms!(device(arch), grid, constants, eos, cfg, uvw..., TS..., pr.pHY′.data, Gⁿ..., forcing)

    calculate_boundary_source_terms!(device(arch), model)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) adams_bashforth_update_source_terms!(device(arch), grid, Gⁿ..., G⁻..., χ)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_source_term_divergence!(device(arch), grid, Guvw..., RHS.data)

    solve_poisson_3d_ppn_gpu_planned!(Tx, Ty, Bx, By, Bz, poisson_solver, grid, RHS, ϕ)
    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) idct_permute!(device(arch), grid, ϕ.data, pr.pNHS.data)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) update_velocities_and_tracers!(device(arch), grid, uvw..., TS..., pr.pNHS.data, Gⁿ..., G⁻..., Δt)

    return nothing
end

"""Store previous source terms before updating them."""
function store_previous_source_terms!(grid::Grid, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gpu[i, j, k] = Gu[i, j, k]
                @inbounds Gpv[i, j, k] = Gv[i, j, k]
                @inbounds Gpw[i, j, k] = Gw[i, j, k]
                @inbounds GpT[i, j, k] = GT[i, j, k]
                @inbounds GpS[i, j, k] = GS[i, j, k]
            end
        end
    end
    @synchronize
end

@inline δρ(ρ₀, βT, T₀, T, i, j, k) = @inbounds -ρ₀ * βT * (T[i, j, k] - T₀)

"Update the hydrostatic pressure perturbation pHY′ and buoyancy δρ."
function update_buoyancy!(grid::Grid, constants, eos, δρ, T, pHY′)
    ρ₀, T₀, βT = eos.ρ₀, eos.T₀, eos.βT
    gΔz = constants.g * grid.Δz

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds δρ[i, j, k] = -ρ₀ * βT * (T[i, j, k] - T₀)

                ∫δρ = (-ρ₀*βT*(T[i, j, 1]-T₀))
                for k′ in 2:k
                    ∫δρ += ((-ρ₀*βT*(T[i, j, k′-1]-T₀)) + (-ρ₀*βT*(T[i, j, k′]-T₀)))
                end
                @inbounds pHY′[i, j, k] = 0.5 * gΔz * ∫δρ
            end
        end
    end

    @synchronize
end

"Store previous value of the source term and calculate current source term."
function calculate_interior_source_terms!(grid::Grid, constants, eos, cfg, u, v, w, T, S, pHY′, Gu, Gv, Gw, GT, GS, F)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    fCor = constants.f
    ρ₀ = eos.ρ₀
    𝜈h, 𝜈v, κh, κv = cfg.𝜈h, cfg.𝜈v, cfg.κh, cfg.κv

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # u-momentum equation
                @inbounds Gu[i, j, k] = (-u∇u(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + fCor*avg_xy(v, Nx, Ny, i, j, k)
                                            - δx_c2f(pHY′, Nx, i, j, k) / (Δx * ρ₀)
                                            + 𝜈∇²u(u, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + F.u(u, v, w, T, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k))

                # v-momentum equation
                @inbounds Gv[i, j, k] = (-u∇v(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            - fCor*avg_xy(u, Nx, Ny, i, j, k)
                                            - δy_c2f(pHY′, Ny, i, j, k) / (Δy * ρ₀)
                                            + 𝜈∇²v(v, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + F.v(u, v, w, T, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k))

                # w-momentum equation: comment about how pressure and buoyancy are handled
                @inbounds Gw[i, j, k] = (-u∇w(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + 𝜈∇²w(w, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + F.w(u, v, w, T, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k))

                # temperature equation
                @inbounds GT[i, j, k] = (-div_flux(u, v, w, T, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + κ∇²(T, κh, κv, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + F.T(u, v, w, T, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k))

                # salinity equation
                @inbounds GS[i, j, k] = (-div_flux(u, v, w, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + κ∇²(S, κh, κv, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                                            + F.S(u, v, w, T, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k))
            end
        end
    end

    @synchronize
end

function adams_bashforth_update_source_terms!(grid::Grid, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS, χ)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds Gu[i, j, k] = (1.5 + χ)*Gu[i, j, k] - (0.5 + χ)*Gpu[i, j, k]
                @inbounds Gv[i, j, k] = (1.5 + χ)*Gv[i, j, k] - (0.5 + χ)*Gpv[i, j, k]
                @inbounds Gw[i, j, k] = (1.5 + χ)*Gw[i, j, k] - (0.5 + χ)*Gpw[i, j, k]
                @inbounds GT[i, j, k] = (1.5 + χ)*GT[i, j, k] - (0.5 + χ)*GpT[i, j, k]
                @inbounds GS[i, j, k] = (1.5 + χ)*GS[i, j, k] - (0.5 + χ)*GpS[i, j, k]
            end
        end
    end
    @synchronize
end

"Store previous value of the source term and calculate current source term."
function calculate_source_term_divergence!(::CPU, grid::Grid, Gu, Gv, Gw, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw).
                @inbounds RHS[i, j, k] = div_f2c(Gu, Gv, Gw, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
            end
        end
    end

    @synchronize
end

function calculate_source_term_divergence!(::GPU, grid::Grid, Gu, Gv, Gw, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw) and applying a permutation which is the first step in the DCT.
                if CUDAnative.ffs(k) == 1  # isodd(k)
                    @inbounds RHS[i, j, convert(UInt32, CUDAnative.floor(k/2) + 1)] = div_f2c(Gu, Gv, Gw, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                else
                    @inbounds RHS[i, j, convert(UInt32, Nz - CUDAnative.floor((k-1)/2))] = div_f2c(Gu, Gv, Gw, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
                end
            end
        end
    end

    @synchronize
end

function idct_permute!(grid::Grid, ϕ, pNHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if k <= Nz/2
                    @inbounds pNHS[i, j, 2k-1] = real(ϕ[i, j, k])
                else
                    @inbounds pNHS[i, j, 2(Nz-k+1)] = real(ϕ[i, j, k])
                end
            end
        end
    end

    @synchronize
end


function update_velocities_and_tracers!(grid::Grid, u, v, w, T, S, pNHS, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS, Δt)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds u[i, j, k] = u[i, j, k] + (Gu[i, j, k] - (δx_c2f(pNHS, Nx, i, j, k) / Δx)) * Δt
                @inbounds v[i, j, k] = v[i, j, k] + (Gv[i, j, k] - (δy_c2f(pNHS, Ny, i, j, k) / Δy)) * Δt
                @inbounds w[i, j, k] = w[i, j, k] + (Gw[i, j, k] - (δz_c2f(pNHS, Nz, i, j, k) / Δz)) * Δt
                @inbounds T[i, j, k] = T[i, j, k] + (GT[i, j, k] * Δt)
                @inbounds S[i, j, k] = S[i, j, k] + (GS[i, j, k] * Δt)
            end
        end
    end

    @synchronize
end


#
# Boundary condition physics specification
#

"Apply boundary conditions by modifying the source term G."
function calculate_boundary_source_terms!(model::Model{A}) where A <: Architecture
    arch = A()

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    Lx, Ly, Lz = model.grid.Lx, model.grid.Ly, model.grid.Lz
    Δx, Δy, Δz = model.grid.Δx, model.grid.Δy, model.grid.Δz

    clock = model.clock
    eos =  model.eos
    cfg = model.configuration
    bcs = model.boundary_conditions
    U = model.velocities
    tr = model.tracers
    G = model.G

    t, iteration = clock.time, clock.iteration
    u, v, w, T, S = U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data
    Gu, Gv, Gw, GT, GS = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    coord = :z #for coord in (:x, :y, :z) when we are ready to support more coordinates.
    𝜈 = cfg.𝜈v
    κ = cfg.κv

    u_x_bcs = getproperty(bcs.u, coord)
    v_x_bcs = getproperty(bcs.v, coord)
    w_x_bcs = getproperty(bcs.w, coord)
    T_x_bcs = getproperty(bcs.T, coord)
    S_x_bcs = getproperty(bcs.S, coord)

    # Apply boundary conditions. We assume there is one molecular 'diffusivity'
    # value, which is passed to apply_bcs.
    apply_bcs!(arch, Val(coord), Bx, By, Bz, u_x_bcs.left, u_x_bcs.right, u, Gu, 𝜈, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # u
    apply_bcs!(arch, Val(coord), Bx, By, Bz, v_x_bcs.left, v_x_bcs.right, v, Gv, 𝜈, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # v
    apply_bcs!(arch, Val(coord), Bx, By, Bz, w_x_bcs.left, w_x_bcs.right, w, Gw, 𝜈, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # w
    apply_bcs!(arch, Val(coord), Bx, By, Bz, T_x_bcs.left, T_x_bcs.right, T, GT, κ, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # T
    apply_bcs!(arch, Val(coord), Bx, By, Bz, S_x_bcs.left, S_x_bcs.right, S, GS, κ, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # S

    return nothing
end

# Do nothing if both boundary conditions are default.
apply_bcs!(::CPU, ::Val{:x}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::CPU, ::Val{:y}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::CPU, ::Val{:z}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing

apply_bcs!(::GPU, ::Val{:x}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::GPU, ::Val{:y}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::GPU, ::Val{:z}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing

# First, dispatch on coordinate.
apply_bcs!(arch, ::Val{:x}, Bx, By, Bz, args...) = @launch device(arch) apply_x_bcs!(args..., threads=(Tx, Ty), blocks=(Bx, By, Bz))
apply_bcs!(arch, ::Val{:y}, Bx, By, Bz, args...) = @launch device(arch) apply_y_bcs!(args..., threads=(Tx, Ty), blocks=(Bx, By, Bz))
apply_bcs!(arch, ::Val{:z}, Bx, By, Bz, args...) = @launch device(arch) apply_z_bcs!(args..., threads=(Tx, Ty), blocks=(Bx, By, Bz))

"Apply a top and/or bottom boundary condition to variable ϕ."
function apply_z_bcs!(top_bc, bottom_bc, ϕ, Gϕ, κ, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz)
    # Loop over i and j to apply a boundary condition on the top.
    @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            apply_z_top_bc!(top_bc, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j)
            apply_z_bottom_bc!(bottom_bc, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j)
        end
    end
    @synchronize
end
