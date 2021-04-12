using Statistics

@kernel function implicit_η!(grid, f, implicit_η_f)
    ### Not sure what to call this
    ### it is for left hand side operator in
    ### (-g∇ₕ² + 1/Δt )ϕⁿ⁺¹ = ϕⁿ / Δt + ∇ₕHUˢᵗᵃʳ

    #
    # g= model.free_surface.gravitational_acceleration
    #
    g = 9.81

    #
    # Δt= simulation.Δt
    #
    Δt = 9.81

    i, j, k = @index(Global, NTuple)
    @inbounds implicit_η_f[i, j] = -g * ∇²ᶜᶜᶜ(i, j, grid, f) + f[i, j] / Δt

    # need this for 2d vertically integrated ∇²hᶜᶜᵃ
end

@kernel function divergence!(grid, u, v, w, div)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function run_pcg_solver_tests(arch)
    Lx, Ly, Lz = 4e6, 6e6, 1
    Nx, Ny, Nz = 100, 150, 1
    grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

    function Amatrix_function!(result, x, arch, grid, bcs; args...)
        event = launch!(arch, grid, :xyz, ∇²!, grid, x, result, dependencies=Event(device(arch)))
        wait(device(arch), event)
        fill_halo_regions!(result, arch)
        return nothing
    end

    # Fields for flow, divergence of flow, RHS, and potential to make non-divergent, ϕ
    velocities = VelocityFields(arch, grid)
    RHS        = CenterField(arch, grid)
    ϕ          = CenterField(arch, grid)

    # Set divergent flow and calculate divergence
    u, v, w  = velocities

    imid = Int(floor(grid.Nx / 2)) + 1
    jmid = Int(floor(grid.Ny / 2)) + 1
    CUDA.@allowscalar u.data[imid, jmid, 1] = 1

    fill_halo_regions!(u, arch)

    event = launch!(arch, grid, :xyz, divergence!, grid, u, v, w, RHS, dependencies=Event(device(arch)))
    wait(device(arch), event)

    fill_halo_regions!(RHS, arch)

    pcg_params = (
        PCmatrix_function = nothing,
        Amatrix_function = Amatrix_function!,
        Template_field = RHS,
        maxit = 1000, # grid.Nx * grid.Ny,
        tol = 1.e-13
    )

    pcg_solver = PreconditionedConjugateGradientSolver(arch = arch, parameters = pcg_params)

    # Set initial guess and solve
    parent(ϕ) .= 0
    @time solve_poisson_equation!(pcg_solver, RHS, ϕ; worda="boo", wordb="cat")

    # Compute ∇² of solution
    result = similar(ϕ)

    event = launch!(arch, grid, :xyz, ∇²!, grid, ϕ, result, dependencies=Event(device(arch)))
    wait(device(arch), event)

    fill_halo_regions!(result, arch)

    CUDA.@allowscalar begin
        @test abs(minimum(result[1:Nx, 1:Ny, 1] .- RHS.data[1:Nx, 1:Ny, 1])) < 1e-12
        @test abs(maximum(result[1:Nx, 1:Ny, 1] .- RHS.data[1:Nx, 1:Ny, 1])) < 1e-12
        @test std(result[1:Nx, 1:Ny, 1] .- RHS.data[1:Nx, 1:Ny, 1]) < 1e-14
    end

    return nothing
end

@testset "Conjugate gradient solvers" begin
    for arch in archs
        @info "Testing conjugate gradient solvers [$(typeof(arch))]..."
        run_pcg_solver_tests(arch)
    end
end
