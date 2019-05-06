@hascuda using CUDAnative, CuArrays

import GPUifyLoops: @launch, @loop, @unroll, @synchronize

using Oceananigans.Operators

const Tx = 16 # CUDA threads per x-block
const Ty = 16 # CUDA threads per y-block

# Increment and decrement integer a with periodic wrapping. So if n == 10 then
# incmod1(11, n) = 1 and decmod1(0, n) = 10.
@inline incmod1(a, n) = ifelse(a==n, 1, a + 1)
@inline decmod1(a, n) = ifelse(a==1, n, a - 1)

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
    dyn_visc = model.d_viscosity

    RHS = model.stepper_tmp.fCC1
    ϕ = model.stepper_tmp.fCC2

    gΔz = model.constants.g * model.grid.Δz
    fCor = model.constants.f

    uvw = U.u.data, U.v.data, U.w.data
    TS = tr.T.data, tr.S.data
    Guvw = G.Gu.data, G.Gv.data, G.Gw.data
    d_visc = dyn_visc.𝜈00.data, dyn_visc.𝜈12.data, dyn_visc.𝜈13.data, dyn_visc.𝜈23.data

    # Source terms at current (Gⁿ) and previous (G⁻) time steps.
    Gⁿ = G.Gu.data, G.Gv.data, G.Gw.data, G.GT.data, G.GS.data
    G⁻ = Gp.Gu.data, Gp.Gv.data, Gp.Gw.data, Gp.GT.data, Gp.GS.data

    Bx, By, Bz = floor(Int, Nx/Tx), floor(Int, Ny/Ty), Nz  # Blocks in grid

    tb = (threads=(Tx, Ty), blocks=(Bx, By, Bz))

    for n in 1:Nt
        χ = ifelse(model.clock.iteration == 0, -0.5, 0.125) # Adams-Bashforth (AB2) parameter.

        @launch device(arch) store_previous_source_terms!(grid, Gⁿ..., G⁻..., threads=(Tx, Ty), blocks=(Bx, By, Bz))
        @launch device(arch) update_buoyancy!(grid, constants, eos, tr.T.data, pr.pHY′.data, threads=(Tx, Ty), blocks=(Bx, By))
        @launch device(arch) calculate_dynamical_viscosity!(grid, constants, eos, cfg, uvw..., d_visc..., threads=(Tx, Ty), blocks=(Bx, By, Bz))
        @launch device(arch) calculate_interior_source_terms!(grid, constants, eos, cfg, uvw..., TS..., pr.pHY′.data, d_visc..., Gⁿ..., forcing, threads=(Tx, Ty), blocks=(Bx, By, Bz))
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

"Update the hydrostatic pressure perturbation pHY′ and buoyancy δρ."
function update_buoyancy!(grid::Grid, constants, eos, T, pHY′)
    gΔz = constants.g * grid.Δz

    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds pHY′[i, j, 1] = 0.5 * gΔz * δρ(eos, T, i, j, 1)
            @unroll for k in 2:grid.Nz
                @inbounds pHY′[i, j, k] = pHY′[i, j, k-1] + gΔz * 0.5 * (δρ(eos, T, i, j, k-1) + δρ(eos, T, i, j, k))
            end
        end
    end

    @synchronize
end

"Store previous value of the source term and calculate current source term."
function calculate_interior_source_terms!(grid::Grid, constants, eos, cfg, u, v, w, T, S, pHY′, 𝜈00, 𝜈12, 𝜈13, 𝜈23, Gu, Gv, Gw, GT, GS, F)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

    prandtl_number = 1
    #- note: prandtl_number should be a model parameter
    Pr_num = prandtl_number

    fCor = constants.f
    ρ₀ = eos.ρ₀
    𝜈h, 𝜈v, κh, κv = cfg.𝜈h, cfg.𝜈v, cfg.κh, cfg.κv

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # u-momentum equation
                @inbounds Gu[i, j, k] = (-u∇u(grid, u, v, w, i, j, k)
                                            + fCor*avg_xy(grid, v, i, j, k)
                                            - δx_c2f(grid, pHY′, i, j, k) / (Δx * ρ₀)
                                            + 𝜈∇²u(grid, u, 𝜈h, 𝜈v, i, j, k)
                                            + gU_visc(grid, u, v, w, 𝜈00, 𝜈12, 𝜈13, i, j, k)
                                            + F.u(grid, u, v, w, T, S, i, j, k))

                # v-momentum equation
                @inbounds Gv[i, j, k] = (-u∇v(grid, u, v, w, i, j, k)
                                            - fCor*avg_xy(grid, u, i, j, k)
                                            - δy_c2f(grid, pHY′, i, j, k) / (Δy * ρ₀)
                                            + 𝜈∇²v(grid, v, 𝜈h, 𝜈v, i, j, k)
                                            + gV_visc(grid, u, v, w, 𝜈00, 𝜈12, 𝜈23, i, j, k)
                                            + F.v(grid, u, v, w, T, S, i, j, k))

                # w-momentum equation: comment about how pressure and buoyancy are handled
                @inbounds Gw[i, j, k] = (-u∇w(grid, u, v, w, i, j, k)
                                            + 𝜈∇²w(grid, w, 𝜈h, 𝜈v, i, j, k)
                                            + gW_visc(grid, u, v, w, 𝜈00, 𝜈13, 𝜈23, i, j, k)
                                            + F.w(grid, u, v, w, T, S, i, j, k))

                # temperature equation
                @inbounds GT[i, j, k] = (-div_flux(grid, u, v, w, T, i, j, k)
                                            + κ∇²(grid, T, κh, κv, i, j, k)
                                            + gTr_diff(grid, T, 𝜈00, Pr_num, i, j, k)
                                            + F.T(grid, u, v, w, T, S, i, j, k))

                # salinity equation
                @inbounds GS[i, j, k] = (-div_flux(grid, u, v, w, S, i, j, k)
                                            + κ∇²(grid, S, κh, κv, i, j, k)
                                            + gTr_diff(grid, S, 𝜈00, Pr_num, i, j, k)
                                            + F.S(grid, u, v, w, T, S, i, j, k))
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
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw).
                @inbounds RHS[i, j, k] = div_f2c(grid, Gu, Gv, Gw, i, j, k)
            end
        end
    end

    @synchronize
end

function calculate_source_term_divergence!(::GPU, grid::Grid, Gu, Gv, Gw, RHS)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                # Calculate divergence of the RHS source terms (Gu, Gv, Gw) and applying a permutation
                # which is the first step in the DCT.
                if CUDAnative.ffs(k) == 1  # isodd(k)
                    @inbounds RHS[i, j, convert(UInt32, CUDAnative.floor(k/2) + 1)] = div_f2c(grid, Gu, Gv, Gw, i, j, k)
                else
                    @inbounds RHS[i, j, convert(UInt32, Nz - CUDAnative.floor((k-1)/2))] = div_f2c(grid, Gu, Gv, Gw, i, j, k)
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
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds u[i, j, k] = u[i, j, k] + (Gu[i, j, k] - (δx_c2f(grid, pNHS, i, j, k) / grid.Δx)) * Δt
                @inbounds v[i, j, k] = v[i, j, k] + (Gv[i, j, k] - (δy_c2f(grid, pNHS, i, j, k) / grid.Δy)) * Δt
                @inbounds w[i, j, k] = w[i, j, k] + (Gw[i, j, k] - (δz_c2f(grid, pNHS, i, j, k) / grid.Δz)) * Δt
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

    grid = model.grid
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
    apply_bcs!(arch, Val(coord), Bx, By, Bz, u_x_bcs.left, u_x_bcs.right, u, Gu, 𝜈, u, v, w, T, S, t, iteration, grid)
    apply_bcs!(arch, Val(coord), Bx, By, Bz, v_x_bcs.left, v_x_bcs.right, v, Gv, 𝜈, u, v, w, T, S, t, iteration, grid)
    apply_bcs!(arch, Val(coord), Bx, By, Bz, w_x_bcs.left, w_x_bcs.right, w, Gw, 𝜈, u, v, w, T, S, t, iteration, grid)
    apply_bcs!(arch, Val(coord), Bx, By, Bz, T_x_bcs.left, T_x_bcs.right, T, GT, κ, u, v, w, T, S, t, iteration, grid)
    apply_bcs!(arch, Val(coord), Bx, By, Bz, S_x_bcs.left, S_x_bcs.right, S, GS, κ, u, v, w, T, S, t, iteration, grid)

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
apply_bcs!(arch, ::Val{:z}, Bx, By, Bz, args...) = @launch device(arch) apply_z_bcs!(args..., threads=(Tx, Ty), blocks=(Bx, By))

"Apply a top and/or bottom boundary condition to variable ϕ. Note that this kernel
MUST be launched with blocks=(Bx, By). If launched with blocks=(Bx, By, Bz), the
boundary condition will be applied Bz times!"
function apply_z_bcs!(top_bc, bottom_bc, ϕ, Gϕ, κ, u, v, w, T, S, t, iteration, grid::RegularCartesianGrid)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            apply_z_top_bc!(top_bc, ϕ, Gϕ, κ, t, grid, u, v, w, T, S, iteration, i, j)
            apply_z_bottom_bc!(bottom_bc, ϕ, Gϕ, κ, t, grid, u, v, w, T, S, iteration, i, j)
        end
    end
    @synchronize
end

"Compute Smagorinsky dynamical viscosity"
function calculate_dynamical_viscosity!(grid::Grid, constants, eos, cfg, u, v, w, 𝜈00, 𝜈12, 𝜈13, 𝜈23 )
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Δx, Δy, Δz = grid.Δx, grid.Δy, grid.Δz

#   smag_coeff = cfg.smag_coefficient
    smag_coeff = 0.1
    fCor = constants.f
    ρ₀ = eos.ρ₀

    s66 = zeros(eltype(u), Nx,Ny,Nz)
    s12 = zeros(eltype(u), Nx,Ny,Nz)
    s13 = zeros(eltype(u), Nx,Ny,Nz)
    s23 = zeros(eltype(u), Nx,Ny,Nz)

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)

                # strain tensor component
                 str11 = δx_f2c(grid, u, i, j, k) / Δx
                 str22 = δy_f2c(grid, v, i, j, k) / Δy
                 str33 = δz_f2c(grid, w, i, j, k) / Δz
                 str12 = ( δy_f2e(grid, u, i, j, k) / Δy + δx_f2e(grid, v, i, j, k) / Δx )*0.5
                 str13 = ( δz_f2e(grid, u, i, j, k) / Δz + δx_f2e(grid, w, i, j, k) / Δx )*0.5
                 str23 = ( δz_f2e(grid, v, i, j, k) / Δz + δy_f2e(grid, w, i, j, k) / Δy )*0.5

                # magnitude of strain tensor: Sum of the square of each component
                s66[i, j, k] = str11*str11 +  str22*str22 +  str33*str33
                s12[i, j, k] = str12*str12
                s13[i, j, k] = str13*str13
                s23[i, j, k] = str23*str23

            end
        end
    end
    @synchronize

    smag_scaled_coeff = smag_coeff * sqrt(2)^3 * (Δx*Δy*Δz)^(2/3)

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)

                # 𝜈00 = sqrt( S11+S22+S33 + 2*(S12+S13+S23) ) @ grid-cell center 
                𝜈00[i, j, k] = smag_scaled_coeff * sqrt( s66[i, j, k]
                   + ( avgx_f2c(grid, s12, i, j, k) + avgx_f2c(grid, s12, i, incmod1(j, Ny), k) )
                   + ( avgz_f2c(grid, s13, i, j, k) + avgz_f2c(grid, s13, incmod1(i, Nx), j, k) )
                   + ( avgz_f2c(grid, s23, i, j, k) + avgz_f2c(grid, s23, i, incmod1(j, Ny), k) )
                                                       )

                # 𝜈12 = sqrt( S11+S22+S33 + 2*(S12+S13+S23) ) @ grid-cell corner
                𝜈12[i, j, k] = smag_scaled_coeff * sqrt( 
                 0.5*( avgx_c2f(grid, s66, i, decmod1(j, Ny), k) + avgx_c2f(grid, s66, i, j, k) )
                   + 2*s12[i, j, k]
                   + ( avgz_f2c(grid, s13, i, decmod1(j, Ny), k) + avgz_f2c(grid, s13, i, j, k) )
                   + ( avgz_f2c(grid, s23, decmod1(i, Nx), j, k) + avgz_f2c(grid, s23, i, j, k) )
                                                       )

                # 𝜈13 = sqrt( S11+S22+S33 + 2*(S12+S13+S23) ) @ above uVel
                𝜈13[i, j, k] = smag_scaled_coeff * sqrt( 
                 0.5*( avgz_c2f(grid, s66, decmod1(i, Nx), j, k) + avgz_c2f(grid, s66, i, j, k) )
                   + ( avgz_c2f(grid, s12, i, j, k) + avgz_c2f(grid, s12, i, incmod1(j, Ny), k) )
                   + 2*s13[i, j, k]
                   + ( avgy_f2c(grid, s23, decmod1(i, Nx), j, k) + avgy_f2c(grid, s23, i, j, k) )
                                                       )

                # 𝜈23 = sqrt( S11+S22+S33 + 2*(S12+S13+S23) ) @ above vVel
                𝜈23[i, j, k] = smag_scaled_coeff * sqrt( 
                 0.5*( avgz_c2f(grid, s66, i, decmod1(j, Ny), k) + avgz_c2f(grid, s66, i, j, k) )
                   + ( avgz_c2f(grid, s12, i, j, k) + avgz_c2f(grid, s12, incmod1(i, Nx), j, k) )
                   + ( avgx_f2c(grid, s13, i, decmod1(j, Ny), k) + avgx_f2c(grid, s13, i, j, k) )
                   + 2*s23[i, j, k]
                                                       )

            end
        end
    end

    @synchronize
end
