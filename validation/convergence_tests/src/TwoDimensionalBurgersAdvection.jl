module TwoDimensionalBurgersAdvection

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Advection: boundary_buffer, VelocityStencil, VorticityStencil, MultiDimensionalScheme
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation, ConservativeFormulation, shallow_water_velocities
using Oceananigans.Fields: interior

using ConvergenceTests: compute_error

# Advection of an isoentropic vortex
# From "Entropy Splitting and Numerical Dissipation", JCP, Yee (2000)

du(x, y, t, μ) = 3/4 - 1/(4 + 4*exp(-4*x+4*y-t)/μ/32)
dv(x, y, t, μ) = 3/4 + 1/(4 + 4*exp(-4*x+4*y-t)/μ/32)

function run_test(; Nx, Δt, stop_iteration, order, U = 0, 
                  architecture = CPU(), topo = (Bounded, Bounded, Flat))
    #####
    ##### Test advection of an isoentropic vortex with a VectorInvariantFormulation and a VorticityStencil
    #####

    μ = 1e-5

    domain = (x=(0, 1), y=(0, 1))
    grid = RectilinearGrid(architecture, topology=topo, size=(Nx, Nx), halo=(6, 6); domain...)

    bcs_u_w =  OpenBoundaryCondition((y, z, t, μ) -> du(0, y, t, μ), parameters = μ)
    bcs_u_e =  OpenBoundaryCondition((y, z, t, μ) -> du(1, y, t, μ), parameters = μ)
    bcs_u_s = ValueBoundaryCondition((x, z, t, μ) -> du(x, 0, t, μ), parameters = μ)
    bcs_u_n = ValueBoundaryCondition((x, z, t, μ) -> du(x, 1, t, μ), parameters = μ)
    
    bcs_v_w = ValueBoundaryCondition((y, z, t, μ) -> dv(0, y, t, μ), parameters = μ)
    bcs_v_e = ValueBoundaryCondition((y, z, t, μ) -> dv(1, y, t, μ), parameters = μ)
    bcs_v_s =  OpenBoundaryCondition((x, z, t, μ) -> dv(x, 0, t, μ), parameters = μ)
    bcs_v_n =  OpenBoundaryCondition((x, z, t, μ) -> dv(x, 1, t, μ), parameters = μ)
    
    u_bcs = FieldBoundaryConditions(west=bcs_u_w, east=bcs_u_e, south=bcs_u_s, north=bcs_u_n)
    v_bcs = FieldBoundaryConditions(west=bcs_v_w, east=bcs_v_e, south=bcs_v_s, north=bcs_v_n)

    model = ShallowWaterModel( grid = grid,
         gravitational_acceleration = 0.0,
                 momentum_advection = WENO(vector_invariant = VelocityStencil(), order = order),
                boundary_conditions = (u = u_bcs, v = v_bcs),
                           coriolis = nothing,
                        formulation = VectorInvariantFormulation())

    set!(model, u = (x, y, z) -> du(x, y, 0, μ),
                v = (x, y, z) -> dv(x, y, 0, μ),
                h = (x, y, z) -> 1.0)

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration)

    @info "Running Isoentropic vortex advection with Ny = $Nx and Δt = $Δt order $order and a VectorInvariantFormulation VorticityStencil..."
    run!(simulation)

    xu = xnodes(model.solution.u)
    yu = ynodes(model.solution.u)
    xv = xnodes(model.solution.v)
    yv = ynodes(model.solution.v)
    u_analytical = zeros(Nx, Nx)
    v_analytical = zeros(Nx, Nx)
    h_analytical = zeros(Nx, Nx)

    for i in 1:Nx, j in 1:Nx
        u_analytical[i, j] += du(xu[i], yu[j], model.clock.time, μ)
        v_analytical[i, j] += dv(xv[i], yv[j], model.clock.time, μ)
        h_analytical[i, j] += 1.0
    end

    # Calculate errors
    uvi_simulation = interior(model.solution.u)[1:Nx, :, 1] |> Array
    uvi_errors = compute_error(uvi_simulation, u_analytical)

    vvi_simulation = interior(model.solution.v)[:, 1:Nx, 1] |> Array
    vvi_errors = compute_error(vvi_simulation, v_analytical)

    hvi_simulation = interior(model.solution.h)[:, :, 1] |> Array
    hvi_errors = compute_error(hvi_simulation, h_analytical)

    #####
    ##### Test advection of an isoentropic vortex with a ConservativeFormulation
    #####

    model = ShallowWaterModel( grid = grid,
         gravitational_acceleration = 0.0,
                 momentum_advection = WENO(order = order),
                boundary_conditions = (uh = u_bcs, vh = v_bcs),
                           coriolis = nothing)

    set!(model, uh = (x, y, z) -> du(x, y, 0, μ),
                vh = (x, y, z) -> dv(x, y, 0, μ),
                h = (x, y, z) -> 1.0)


    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration)

    @info "Running Isoentropic vortex advection with Ny = $Nx and Δt = $Δt order $order and a ConservativeFormulation..."
    run!(simulation)

    u, v = shallow_water_velocities(model)

    # Calculate errors
    ucf_simulation = interior(u)[1:Nx, :, 1] |> Array
    ucf_errors = compute_error(ucf_simulation, u_analytical)

    vcf_simulation = interior(v)[:, 1:Nx, 1] |> Array
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
                           L₁ = uvi_errors.L₁,
                           L∞ = uvi_errors.L∞),

            vvv = (simulation = vvi_simulation,
                   analytical = v_analytical,
                           L₁ = vvi_errors.L₁,
                           L∞ = vvi_errors.L∞),

            hvv = (simulation = hvi_simulation,
                   analytical = h_analytical,
                           L₁ = hvi_errors.L₁,
                           L∞ = hvi_errors.L∞),

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
