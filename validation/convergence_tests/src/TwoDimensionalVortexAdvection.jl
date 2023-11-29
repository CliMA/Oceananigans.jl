module TwoDimensionalVortexAdvection

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Advection: required_halo_size, VelocityStencil
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation, ConservativeFormulation, shallow_water_velocities
using Oceananigans.Fields: interior

using ConvergenceTests: compute_error

# Advection of an isoentropic vortex
# From "Entropy Splitting and Numerical Dissipation", JCP, Yee (2000)

r2(x, y, xᵥ, yᵥ) = (x - xᵥ)^2 + (y - yᵥ)^2
δu(x, y, t, U, β, xᵥ, yᵥ) = - β / 2π * exp(0.5 - 0.5*r2(x - U * t, y, xᵥ, yᵥ)) * (y - yᵥ)
δv(x, y, t, U, β, xᵥ, yᵥ) = + β / 2π * exp(0.5 - 0.5*r2(x - U * t, y, xᵥ, yᵥ)) * (x - xᵥ)
δh(x, y, t, U, β, xᵥ, yᵥ) = (1 - β^2 / 16π^2 * exp(1 - r2(x - U * t, y, xᵥ, yᵥ)))

function run_test(; Nx, Δt, stop_iteration, U = 0, order,
                  architecture = CPU(), topo = (Periodic, Periodic, Flat))

    β  = 1.0         
    xᵥ = 5.0
    yᵥ = 0.0         

    #####
    ##### Test advection of an isoentropic vortex with a VectorInvariantFormulation and a VorticityStencil
    #####

    domain = (x=(0, 10), y=(-5, 5))
    grid = RectilinearGrid(architecture, topology=topo, size=(Nx, Nx), halo=(6, 6); domain...)

    model = ShallowWaterModel( grid = grid,
         gravitational_acceleration = 2.0,
                 momentum_advection = VectorInvariant(vorticity_scheme = WENO(; order)),
                     mass_advection = WENO(; order),
                           coriolis = nothing,
                            closure = nothing,
                        formulation = VectorInvariantFormulation())

    set!(model, u = (x, y, z) -> U + δu(x, y, 0, U, β, xᵥ, yᵥ),
                v = (x, y, z) -> δv(x, y, 0, U, β, xᵥ, yᵥ),
                h = (x, y, z) -> δh(x, y, 0, U, β, xᵥ, yᵥ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=50)

    @info "Running Isoentropic vortex advection with Ny = $Nx and Δt = $Δt order $order and a VectorInvariantFormulation VorticityStencil..."
    run!(simulation)

    xh = xnodes(model.solution.h)
    yh = ynodes(model.solution.h)
    xu = xnodes(model.solution.u)
    yu = ynodes(model.solution.u)
    xv = xnodes(model.solution.v)
    yv = ynodes(model.solution.v)
    u_analytical = zeros(Nx, Nx)
    v_analytical = zeros(Nx, Nx)
    h_analytical = zeros(Nx, Nx)
    for i in 1:Nx, j in 1:Nx
        u_analytical[i, j] = U + δu.(xu[i], yu[j], model.clock.time, U, β, xᵥ, yᵥ)
        v_analytical[i, j] = δv.(xv[i], yv[j], model.clock.time, U, β, xᵥ, yᵥ)
        h_analytical[i, j] = δh.(xh[i], yh[j], model.clock.time, U, β, xᵥ, yᵥ)
    end

    # Calculate errors
    uvi_simulation = interior(model.solution.u)[:, :, 1] |> Array
    uvi_errors = compute_error(uvi_simulation, u_analytical)

    vvi_simulation = interior(model.solution.v)[:, :, 1] |> Array
    vvi_errors = compute_error(vvi_simulation, v_analytical)

    hvi_simulation = interior(model.solution.h)[:, :, 1] |> Array
    hvi_errors = compute_error(hvi_simulation, h_analytical)

    #####
    ##### Test advection of an isoentropic vortex with a ConservativeFormulation
    #####

    model = ShallowWaterModel( grid = grid,
         gravitational_acceleration = 1.0,
                 momentum_advection = WENO(order = order),
                     mass_advection = WENO(order = order),
                           coriolis = nothing,
                            closure = nothing)
    
    set!(model, uh = (x, y, z) -> (U + δu(x, y, 0, U, β, xᵥ, yᵥ)) * δh(x, y, 0, U, β, xᵥ, yᵥ),
                vh = (x, y, z) -> δv(x, y, 0, U, β, xᵥ, yᵥ) * δh(x, y, 0, U, β, xᵥ, yᵥ),
                h  = (x, y, z) -> δh(x, y, 0, U, β, xᵥ, yᵥ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration)

    @info "Running Isoentropic vortex advection with Ny = $Nx and Δt = $Δt order $order and a ConservativeFormulation..."
    run!(simulation)

    u, v = shallow_water_velocities(model)

    # Calculate errors
    # Calculate errors
    ucf_simulation = interior(u)[:, :, 1] |> Array
    ucf_errors = compute_error(ucf_simulation, u_analytical)

    vcf_simulation = interior(v)[:, :, 1] |> Array
    vcf_errors = compute_error(vcf_simulation, v_analytical)

    hcf_simulation = interior(model.solution.h)[:, :, 1] |> Array
    hcf_errors = compute_error(hcf_simulation, h_analytical)

    return (

            uvi = (simulation = uvi_simulation,
                   analytical = u_analytical,
                           L₁ = uvi_errors.L₁,
                           L∞ = uvi_errors.L∞),

            vvi = (simulation = vvi_simulation,
                   analytical = v_analytical,
                           L₁ = vvi_errors.L₁,
                           L∞ = vvi_errors.L∞),

            hvi = (simulation = hvi_simulation,
                   analytical = h_analytical,
                           L₁ = hvi_errors.L₁,
                           L∞ = hvi_errors.L∞),

            uvv = (simulation = uvi_simulation,
                   analytical = u_analytical,
                           L₁ = uvv_errors.L₁,
                           L∞ = uvv_errors.L∞),

            vvv = (simulation = vvi_simulation,
                   analytical = v_analytical,
                           L₁ = vvi_errors.L₁,
                           L∞ = vvi_errors.L∞),

            hvv = (simulation = hvi_simulation,
                   analytical = h_analytical,
                           L₁ = hvv_errors.L₁,
                           L∞ = hvv_errors.L∞),

            ucf = (simulation = ucf_simulation,
                   analytical = u_analytical,
                           L₁ = ucf_errors.L₁,
                           L∞ = ucf_errors.L∞),

            vcf = (simulation = vcf_simulation,
                   analytical = v_analytical,
                           L₁ = vcf_errors.L₁,
                           L∞ = vcf_errors.L∞),

            hcf = (simulation = hcf_simulation,
                   analytical = h_analytical,
                           L₁ = hcf_errors.L₁,
                           L∞ = hcf_errors.L∞),                               
                           
                grid = grid
            )
end

using PyPlot

using Oceananigans.Grids

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

function unpack_errors(results)
    uvi_L₁ = map(r -> r.uvi.L₁, results)
    vvi_L₁ = map(r -> r.vvi.L₁, results)
    hvi_L₁ = map(r -> r.hvi.L₁, results)

    uvv_L₁ = map(r -> r.uvv.L₁, results)
    vvv_L₁ = map(r -> r.vvv.L₁, results)
    hvv_L₁ = map(r -> r.hvv.L₁, results)

    ucf_L₁ = map(r -> r.ucf.L₁, results)
    vcf_L₁ = map(r -> r.vcf.L₁, results)
    hcf_L₁ = map(r -> r.hcf.L₁, results)
    
    uvi_L∞ = map(r -> r.uvi.L∞, results)
    vvi_L∞ = map(r -> r.vvi.L∞, results)
    hvi_L∞ = map(r -> r.hvi.L∞, results)

    uvv_L∞ = map(r -> r.uvv.L∞, results)
    vvv_L∞ = map(r -> r.vvv.L∞, results)
    hvv_L∞ = map(r -> r.hvv.L∞, results)

    ucf_L∞ = map(r -> r.ucf.L∞, results)
    vcf_L∞ = map(r -> r.vcf.L∞, results)
    hcf_L∞ = map(r -> r.hcf.L∞, results)

    return (
        uvi_L₁,
        vvi_L₁,
        hvi_L₁,

        uvv_L₁,
        vvv_L₁,
        hvv_L₁,
        
        ucf_L₁,
        vcf_L₁,
        hcf_L₁,
        
        uvi_L∞,
        vvi_L∞,
        hvi_L∞,
        
        uvv_L∞,
        vvv_L∞,
        hvv_L∞,

        ucf_L∞,
        vcf_L∞,
        hcf_L∞,
        )
end

end # module
