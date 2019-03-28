@hascuda using GPUifyLoops, CUDAnative, CuArrays

using Oceananigans.Operators

const Tx = 16 # Threads per x-block
const Ty = 16 # Threads per y-block

"""
    time_step!(model, Nt, Δt)

Step forward `model` `Nt` time steps using a second-order Adams-Bashforth
method with step size `Δt`.
"""
function time_step!(model::Model{arch}, Nt, Δt) where arch <: Architecture
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

    for n in 1:Nt
        # Adams-Bashforth (AB2) parameter.
        χ = ifelse(model.clock.iteration == 0, -0.5, 0.125)

        time_step_kernels!(arch(), model, χ, Δt)

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

    store_previous_source_terms!(device(arch), grid, Gⁿ..., G⁻...)

    update_buoyancy!(device(arch), grid, constants, eos, δρ.data, tr.T.data, pr.pHY′.data)

    calculate_interior_source_terms!(device(arch), grid, constants, eos, cfg, uvw..., TS..., pr.pHY′.data, Gⁿ..., forcing)

    calculate_boundary_source_terms!(device(arch), 0, 0, 0, bcs, eos.ρ₀, cfg.κh, cfg.κv, cfg.𝜈h, cfg.𝜈v,
                               clock.time, clock.iteration, Nx, Ny, Nz, Lx, Ly, Lz, Δx, Δy, Δz,
                               U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data,
                               G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data)

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

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) store_previous_source_terms!(device(arch), grid, Gⁿ..., G⁻...)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) update_buoyancy!(device(arch), grid, constants, eos, δρ.data, tr.T.data, pr.pHY′.data)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_interior_source_terms!(device(arch), grid, constants, eos, cfg, uvw..., TS..., pr.pHY′.data, Gⁿ..., forcing)

    calculate_boundary_source_terms!(device(arch), Bx, By, Bz, bcs, eos.ρ₀, cfg.κh, cfg.κv, cfg.𝜈h, cfg.𝜈v,
                               clock.time, clock.iteration, Nx, Ny, Nz, Lx, Ly, Lz, Δx, Δy, Δz,
                               U.u.data, U.v.data, U.w.data, tr.T.data, tr.S.data,
                               G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) adams_bashforth_update_source_terms!(device(arch), grid, Gⁿ..., G⁻..., χ)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) calculate_source_term_divergence!(device(arch), grid, Guvw..., RHS.data)

    solve_poisson_3d_ppn_gpu_planned!(Tx, Ty, Bx, By, Bz, poisson_solver, grid, RHS, ϕ)
    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) idct_permute!(device(arch), grid, ϕ.data, pr.pNHS.data)

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) update_velocities_and_tracers!(device(arch), grid, uvw..., TS..., pr.pNHS.data, Gⁿ..., G⁻..., Δt)

    return nothing
end

"""Store previous source terms before updating them."""
function store_previous_source_terms!(::Val{Dev}, grid::Grid, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS) where Dev
    @setup Dev

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
function update_buoyancy!(::Val{Dev}, grid::Grid, constants, eos, δρ, T, pHY′) where Dev
    @setup Dev

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
function calculate_interior_source_terms!(::Val{Dev}, grid::Grid, constants, eos, cfg,
    u, v, w, T, S, pHY′, Gu, Gv, Gw, GT, GS, F) where Dev

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    fCor = constants.f
    ρ₀ = eos.ρ₀
    𝜈h, 𝜈v, κh, κv = cfg.𝜈h, cfg.𝜈v, cfg.κh, cfg.κv

    @setup Dev

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

function adams_bashforth_update_source_terms!(::Val{Dev}, grid::Grid, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS, χ) where Dev
    @setup Dev

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
function calculate_source_term_divergence!(::Val{:CPU}, grid::Grid, Gu, Gv, Gw, RHS)
    @setup :CPU

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

function calculate_source_term_divergence!(::Val{:GPU}, grid::Grid, Gu, Gv, Gw, RHS)
    @setup :GPU

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

function idct_permute!(::Val{Dev}, grid::Grid, ϕ, pNHS) where Dev
    @setup Dev

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if k <= grid.Nz/2
                    @inbounds pNHS[i, j, 2k-1] = real(ϕ[i, j, k])
                else
                    @inbounds pNHS[i, j, 2(Nz-k+1)] = real(ϕ[i, j, k])
                end
            end
        end
    end

    @synchronize
end


function update_velocities_and_tracers!(::Val{Dev}, grid::Grid, u, v, w, T, S, pNHS, Gu, Gv, Gw, GT, GS, Gpu, Gpv, Gpw, GpT, GpS, Δt) where Dev
    @setup Dev

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
function calculate_boundary_source_terms!(Dev, Bx, By, Bz, bcs, ρ₀, κh, κv, 𝜈h, 𝜈v,
                                          t, iteration, Nx, Ny, Nz, Lx, Ly, Lz, Δx, Δy, Δz,
                                          u, v, w, T, S, Gu, Gv, Gw, GT, GS)

    coord = :z #for coord in (:x, :y, :z) when we are ready to support more coordinates.
    𝜈 = 𝜈v
    κ = κv

    u_x_bcs = getproperty(bcs.u, coord)
    v_x_bcs = getproperty(bcs.v, coord)
    w_x_bcs = getproperty(bcs.w, coord)
    T_x_bcs = getproperty(bcs.T, coord)
    S_x_bcs = getproperty(bcs.S, coord)

    # Apply boundary conditions. We assume there is one molecular 'diffusivity'
    # value, which is passed to apply_bcs.
    apply_bcs!(Dev, Val(coord), Bx, By, Bz, u_x_bcs.left, u_x_bcs.right, u, Gu, 𝜈, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # u
    apply_bcs!(Dev, Val(coord), Bx, By, Bz, v_x_bcs.left, v_x_bcs.right, v, Gv, 𝜈, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # v
    apply_bcs!(Dev, Val(coord), Bx, By, Bz, w_x_bcs.left, w_x_bcs.right, w, Gw, 𝜈, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # w
    apply_bcs!(Dev, Val(coord), Bx, By, Bz, T_x_bcs.left, T_x_bcs.right, T, GT, κ, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # T
    apply_bcs!(Dev, Val(coord), Bx, By, Bz, S_x_bcs.left, S_x_bcs.right, S, GS, κ, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) # S

    return nothing
end

# Do nothing if both boundary conditions are default.
apply_bcs!(::Val{:CPU}, ::Val{:x}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::Val{:CPU}, ::Val{:y}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::Val{:CPU}, ::Val{:z}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing

apply_bcs!(::Val{:GPU}, ::Val{:x}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::Val{:GPU}, ::Val{:y}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing
apply_bcs!(::Val{:GPU}, ::Val{:z}, Bx, By, Bz, left_bc::BC{<:Default, T}, right_bc::BC{<:Default, T}, args...) where {T} = nothing


# First, dispatch on coordinate.
apply_bcs!(::Val{:CPU}, ::Val{:x}, Bx, By, Bz, args...) = apply_x_bcs!(Val(:CPU), args...)
apply_bcs!(::Val{:CPU}, ::Val{:y}, Bx, By, Bz, args...) = apply_y_bcs!(Val(:CPU), args...)
apply_bcs!(::Val{:CPU}, ::Val{:z}, Bx, By, Bz, args...) = apply_z_bcs!(Val(:CPU), args...)

apply_bcs!(::Val{:GPU}, ::Val{:x}, Bx, By, Bz, args...) = @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) apply_x_bcs!(Val(:GPU), args...)
apply_bcs!(::Val{:GPU}, ::Val{:y}, Bx, By, Bz, args...) = @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) apply_y_bcs!(Val(:GPU), args...)
apply_bcs!(::Val{:GPU}, ::Val{:z}, Bx, By, Bz, args...) = @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) apply_z_bcs!(Val(:GPU), args...)

#
# Physics goes here.
#

#=
Currently we support flux and gradient boundary conditions
at the top and bottom of the domain.

Notes:

- The boundary condition on a z-boundary is a callable object with arguments

      (t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j),

  where i and j are the x and y indices, respectively. No other function signature will work.
  We do not have abstractions that generalize to non-uniform grids.

- We assume that the boundary tendency has been previously calculated for
  a 'no-flux' boundary condition.

  This means that boudnary conditions take the form of
  an addition/subtraction to the tendency associated with a flux at point (A, A, I) below the bottom cell.
  This paradigm holds as long as consider boundary conditions on (A, A, C) variables only, where A is
  "any" of C or I.

 - We use the physics-based convention that

        flux = -κ * gradient,

    and that

        tendency = ∂ϕ/∂t = Gϕ = - ∇ ⋅ flux

=#

# Do nothing in default case. These functions are called in cases where one of the
# z-boundaries is set, but not the other.
@inline apply_z_top_bc!(args...) = nothing
@inline apply_z_bottom_bc!(args...) = nothing

# These functions compute vertical fluxes for (A, A, C) quantities. They are not currently used.
@inline ∇κ∇ϕ_t(κ, ϕt, ϕt₋₁, flux, Δzc, Δzf) = (      -flux        - κ*(ϕt - ϕt₋₁)/Δzc ) / Δzf
@inline ∇κ∇ϕ_b(κ, ϕb, ϕb₊₁, flux, Δzc, Δzf) = ( κ*(ϕb₊₁ - ϕb)/Δzc +       flux        ) / Δzf

"Add flux divergence to ∂ϕ/∂t associated with a top boundary condition on ϕ."
@inline apply_z_top_bc!(top_flux::BC{<:Flux, <:Function}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, 1] -= top_flux.condition(t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j) / Δz

@inline apply_z_top_bc!(top_flux::BC{<:Flux, <:Number}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, 1] -= top_flux.condition / Δz

@inline apply_z_top_bc!(top_flux::BC{<:Flux, <:AbstractArray}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, 1] -= top_flux.condition[i, j] / Δz

@inline apply_z_top_bc!(top_gradient::BC{<:Gradient, <:Function}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, 1] += κ*top_gradient.condition(t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j) / Δz

@inline apply_z_top_bc!(top_gradient::BC{<:Gradient, <:Number}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, 1] += κ*top_gradient.condition / Δz

@inline apply_z_top_bc!(top_gradient::BC{<:Gradient, <:AbstractArray}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, 1] += κ*top_gradient.condition[i, j] / Δz

"Add flux divergence to ∂ϕ/∂t associated with a bottom boundary condition on ϕ."
@inline apply_z_bottom_bc!(bottom_flux::BC{<:Flux, <:Function}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, Nz] += bottom_flux.condition(t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j) / Δz

@inline apply_z_bottom_bc!(bottom_flux::BC{<:Flux, <:Number}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, Nz] += bottom_flux.condition / Δz

@inline apply_z_bottom_bc!(bottom_flux::BC{<:Flux, <:AbstractArray}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, Nz] += bottom_flux.condition[i, j] / Δz

@inline apply_z_bottom_bc!(bottom_gradient::BC{<:Gradient, <:Function}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, Nz] -= κ*bottom_gradient.condition(t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j) / Δz

@inline apply_z_bottom_bc!(bottom_gradient::BC{<:Gradient, <:Number}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, Nz] -= κ*bottom_gradient.condition / Δz

@inline apply_z_bottom_bc!(bottom_gradient::BC{<:Gradient, <:AbstractArray}, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S,
    iteration, i, j) = Gϕ[i, j, Nz] -= κ*bottom_gradient.condition[i, j] / Δz

"Apply a top and/or bottom boundary condition to variable ϕ."
function apply_z_bcs!(::Val{Dev}, top_bc, bottom_bc, ϕ, Gϕ, κ, u, v, w, T, S, t, iteration, Nx, Ny, Nz, Δx, Δy, Δz) where Dev
    @setup Dev

    # Loop over i and j to apply a boundary condition on the top.
    @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            apply_z_top_bc!(top_bc, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j)
            apply_z_bottom_bc!(bottom_bc, ϕ, Gϕ, κ, t, Δx, Δy, Δz, Nx, Ny, Nz, u, v, w, T, S, iteration, i, j)
        end
    end

    return nothing
end
