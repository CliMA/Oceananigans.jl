using Oceananigans.Solvers: solve_for_pressure!, solve_poisson_equation!

function pressure_solver_instantiates(FT, Nx, Ny, Nz, planner_flag)
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(100, 100, 100))
    solver = PressureSolver(CPU(), grid, PressureBoundaryConditions(grid), planner_flag)
    return true  # Just making sure the PressureSolver does not error/crash.
end

function divergence_free_poisson_solution(arch, FT, topology, Nx, Ny, Nz, planner_flag=FFTW.MEASURE)
    ArrayType = array_type(arch)
    grid = RegularCartesianGrid(FT, topology=topology, size=(Nx, Ny, Nz), extent=(1.0, 2.5, π))
    fbcs = TracerBoundaryConditions(grid)
    pbcs = PressureBoundaryConditions(grid)
    solver = PressureSolver(arch, grid, fbcs, planner_flag)

    # Generate right hand side from a random (divergent) velocity field.
    Ru = CellField(FT, arch, grid, UVelocityBoundaryConditions(grid))
    Rv = CellField(FT, arch, grid, VVelocityBoundaryConditions(grid))
    Rw = CellField(FT, arch, grid, WVelocityBoundaryConditions(grid))
    U = (u=Ru, v=Rv, w=Rw)

    set!(Ru, rand(Nx, Ny, Nz))
    set!(Rv, rand(Nx, Ny, Nz))
    set!(Rw, rand(Nx, Ny, Nz))

    # Adding (nothing, nothing) in case we need to dispatch on ::NFBC
    fill_halo_regions!(Ru, arch, nothing, nothing)
    fill_halo_regions!(Rv, arch, nothing, nothing)
    fill_halo_regions!(Rw, arch, nothing, nothing)

    # Compute the right hand side R = ∇⋅U
    R = zeros(Nx, Ny, Nz) |> ArrayType
    event = launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R,
                    dependencies=Event(device(arch)))
    wait(device(arch), event)

    ϕ   = CellField(FT, arch, grid, pbcs)  # "pressure"
    ∇²ϕ = CellField(FT, arch, grid, pbcs)

    # Using Δt = 1 but it doesn't matter since velocities = 0.
    solve_for_pressure!(ϕ.data, solver, arch, grid, 1, datatuple(U))

    fill_halo_regions!(ϕ, arch)
    event = launch!(arch, grid, :xyz, ∇²!, grid, ϕ.data, ∇²ϕ.data, dependencies=Event(device(arch)))
    wait(device(arch), event)
    fill_halo_regions!(∇²ϕ, arch)

    return CUDA.@allowscalar interior(∇²ϕ) ≈ R
end

#####
##### Test that Poisson solver error converges as error ~ N⁻²
#####

ψ(::Type{Bounded}, n, x) = cos(n*x/2)
ψ(::Type{Periodic}, n, x) = cos(n*x)

k²(::Type{Bounded}, n) = (n/2)^2
k²(::Type{Periodic}, n) = n^2

function analytical_poisson_solver_test(arch, N, topo; FT=Float64, mode=1)
    grid = RegularCartesianGrid(FT, topology=topo, size=(N, N, N), x=(0, 2π), y=(0, 2π), z=(0, 2π))
    solver = PressureSolver(arch, grid, TracerBoundaryConditions(grid))

    xC, yC, zC = nodes((Cell, Cell, Cell), grid, reshape=true)

    Tx, Ty, Tz = topology(grid)
    Ψ(x, y, z) = ψ(Tx, mode, x) * ψ(Ty, mode, y) * ψ(Tz, mode, z)
    f(x, y, z) = -(k²(Tx, mode) + k²(Ty, mode) + k²(Tz, mode)) * Ψ(x, y, z)

    @. solver.storage = f(xC, yC, zC)
    solve_poisson_equation!(solver, grid)
    ϕ = real.(solver.storage)

    L¹_error = mean(abs, ϕ - Ψ.(xC, yC, zC))

    return L¹_error
end

function poisson_solver_convergence(arch, topo, N¹, N²; FT=Float64)
    error¹ = analytical_poisson_solver_test(arch, N¹, topo; FT=FT)
    error² = analytical_poisson_solver_test(arch, N², topo; FT=FT)

    rate = log(error¹ / error²) / log(N² / N¹)

    Tx, Ty, Tz = topo
    @info "Convergence of L¹-normed error, $FT, ($(N¹)³ -> $(N²)³), topology=($Tx, $Ty, $Tz): $rate"

    return isapprox(rate, 2, rtol=5e-3)
end

#####
##### Run pressure solver tests
#####

const PPP_topo = (Periodic, Periodic, Periodic)
const PPB_topo = (Periodic, Periodic, Bounded)
const PBB_topo = (Periodic, Bounded,  Bounded)
const BBB_topo = (Bounded,  Bounded,  Bounded)

@testset "Pressure solvers" begin
    @info "Testing pressure solvers..."

    @testset "Pressure solver instantiation" begin
        @info "  Testing pressure solver instantiation..."

        for FT in float_types
            @test pressure_solver_instantiates(FT, 32, 32, 32, FFTW.ESTIMATE)
            @test pressure_solver_instantiates(FT, 1,  32, 32, FFTW.MEASURE)
            @test pressure_solver_instantiates(FT, 32,  1, 32, FFTW.ESTIMATE)
            @test pressure_solver_instantiates(FT,  1,  1, 32, FFTW.MEASURE)
        end
    end

    @testset "Divergence-free solution [CPU]" begin
        @info "  Testing divergence-free solution [CPU]..."

        for topo in (PPP_topo, PPB_topo, PBB_topo, BBB_topo)
            @info "    Testing $topo topology on square grids..."
            for N in [7, 16], FT in float_types
                @test divergence_free_poisson_solution(CPU(), FT, topo, N, N, N, FFTW.ESTIMATE)
                @test divergence_free_poisson_solution(CPU(), FT, topo, 1, N, N, FFTW.MEASURE)
                @test divergence_free_poisson_solution(CPU(), FT, topo, N, 1, N, FFTW.ESTIMATE)
                @test divergence_free_poisson_solution(CPU(), FT, topo, 1, 1, N, FFTW.MEASURE)
            end
        end

        Ns = [11, 16]
        for topo in (PPP_topo, PPB_topo, PBB_topo, BBB_topo)
            @info "    Testing $topo topology on rectangular grids..."
            for Nx in Ns, Ny in Ns, Nz in Ns, FT in float_types
                @test divergence_free_poisson_solution(CPU(), FT, topo, Nx, Ny, Nz, FFTW.ESTIMATE)
            end
        end
    end

    @hascuda @testset "Divergence-free solution [GPU]" begin
        @info "  Testing divergence-free solution [GPU]..."
        for topo in (PPP_topo, PPB_topo, PBB_topo)
            @info "    Testing $topo topology on GPUs..."
            @test divergence_free_poisson_solution(GPU(), Float64, topo, 16, 16, 16)
            @test divergence_free_poisson_solution(GPU(), Float64, topo, 32, 32, 32)
            @test divergence_free_poisson_solution(GPU(), Float64, topo, 32, 32, 16)
            @test divergence_free_poisson_solution(GPU(), Float64, topo, 16, 32, 24)
        end
    end

    @testset "Convergence to analytical solution" begin
        @info "  Testing convergence to analytical solution..."
        @test poisson_solver_convergence(CPU(), (Periodic, Periodic, Periodic), 2^6, 2^7)
        @test poisson_solver_convergence(CPU(), (Periodic, Periodic,  Bounded), 2^6, 2^7)
        @test poisson_solver_convergence(CPU(), (Periodic,  Bounded,  Bounded), 2^6, 2^7)
        @test poisson_solver_convergence(CPU(), (Bounded,   Bounded,  Bounded), 2^6, 2^7)
    end

end
