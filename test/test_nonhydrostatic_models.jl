include("dependencies_for_runtests.jl")

using Oceananigans.Grids: required_halo_size_x, required_halo_size_y, required_halo_size_z
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FreeSurfaceLaplacian,
                            DeflatedFourierTridiagonalPreconditioner,
                            fft_free_surface_preconditioner, no_gauge_enforcement!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models.NonhydrostaticModels: nonhydrostatic_pressure_solver

@testset "Models" begin
    @info "Testing models..."

    grids = (RectilinearGrid(CPU(), size=(1, 1, 1), extent=(1, 1, 1)),
             LatitudeLongitudeGrid(CPU(), size=(1, 1, 1), longitude=(-180, 180), latitude=(-20, 20), z=(-1, 0)))

    for grid in grids
        @testset "$grid grid construction" begin
            @info "  Testing $grid grid construction..."
                @test_throws TypeError NonhydrostaticModel(grid; boundary_conditions=1)
                @test_throws TypeError NonhydrostaticModel(grid; forcing=2)
                @test_throws TypeError NonhydrostaticModel(grid; background_fields=3)

        end
    end

    topos = ((Periodic, Periodic, Periodic),
             (Periodic, Periodic,  Bounded),
             (Periodic,  Bounded,  Bounded),
             (Bounded,   Bounded,  Bounded))

    for topo in topos
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
                arch isa GPU && topo == (Bounded, Bounded, Bounded) && continue

                grid = RectilinearGrid(arch, FT, topology=topo, size=(16, 16, 2), extent=(1, 2, 3))
                model = NonhydrostaticModel(grid)

                @test model isa NonhydrostaticModel
            end
        end
    end

    @testset "Adjustment of halos in NonhydrostaticModel constructor" begin
        @info "  Testing adjustment of halos in NonhydrostaticModel constructor..."

        minimal_grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 2, 3), halo=(1, 1, 1))
          funny_grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 2, 3), halo=(1, 3, 4))

        # Model ensures that halos are at least of size 1
        model = NonhydrostaticModel(minimal_grid)
        @test model.grid.Hx == 1 && model.grid.Hy == 1 && model.grid.Hz == 1

        model = NonhydrostaticModel(funny_grid)
        @test model.grid.Hx == 1 && model.grid.Hy == 3 && model.grid.Hz == 4

        # Model ensures that halos are at least of size 2
        for scheme in (Centered(order=4), UpwindBiased(order=3))
            model = NonhydrostaticModel(minimal_grid; advection=scheme)
            @test model.grid.Hx == 2 && model.grid.Hy == 2 && model.grid.Hz == 2

            model = NonhydrostaticModel(funny_grid; advection=scheme)
            @test model.grid.Hx == 2 && model.grid.Hy == 3 && model.grid.Hz == 4
        end

        # Model ensures that halos are at least of size 3
        for scheme in (WENO(), UpwindBiased(order=5))
            model = NonhydrostaticModel(minimal_grid; advection=scheme)
            @test model.grid.Hx == 3 && model.grid.Hy == 3 && model.grid.Hz == 3

            model = NonhydrostaticModel(funny_grid; advection=scheme)
            @test model.grid.Hx == 3 && model.grid.Hy == 3 && model.grid.Hz == 4
        end

        # Model ensures that halos are at least of size 2 with ScalarBiharmonicDiffusivity
        model = NonhydrostaticModel(minimal_grid; closure=ScalarBiharmonicDiffusivity())
        @test model.grid.Hx == 2 && model.grid.Hy == 2 && model.grid.Hz == 2

        model = NonhydrostaticModel(funny_grid; closure=ScalarBiharmonicDiffusivity())
        @test model.grid.Hx == 2 && model.grid.Hy == 3 && model.grid.Hz == 4

        @info "  Testing adjustment of advection schemes in NonhydrostaticModel constructor..."
        small_grid = RectilinearGrid(size=(4, 2, 4), extent=(1, 2, 3), halo=(1, 1, 1))

        # These tests are broken at the moment, because limiting does not work as intended:
        # It is not enough to limit the advection scheme in one direction since the same scheme
        # (only for momentum advection) is used to reconstruct the advecting velocity in the
        # tangential direction, leading to an out-of-bounds access:
        # for example, if the grid is 4, 2, 4 and we limit advection in y to Upwind(3), still we
        # will have Upwind(7) in x that needs to compute the advecting velocity (for example u for v advection)
        # using an 8-point stencil, thus leading to an out-of-bounds error
        # Model ensures that halos are at least of size 1
        # See issue
        # model = NonhydrostaticModel(small_grid, advection=WENO())
        # @test model.advection isa FluxFormAdvection
        # @test required_halo_size_x(model.advection) == 3
        # @test required_halo_size_y(model.advection) == 2
        # @test required_halo_size_z(model.advection) == 3

        # model = NonhydrostaticModel(small_grid, advection=UpwindBiased(; order = 9))
        # @test model.advection isa FluxFormAdvection
        # @test required_halo_size_x(model.advection) == 4
        # @test required_halo_size_y(model.advection) == 2
        # @test required_halo_size_z(model.advection) == 4

        # model = NonhydrostaticModel(small_grid, advection=Centered(; order = 10))
        # @test model.advection isa FluxFormAdvection
        # @test required_halo_size_x(model.advection) == 4
        # @test required_halo_size_y(model.advection) == 2
        # @test required_halo_size_z(model.advection) == 4
    end

    @testset "Model construction with single tracer and nothing tracer" begin
        @info "  Testing model construction with single tracer and nothing tracer..."
        for arch in archs
            for grid in grids
                model = NonhydrostaticModel(grid; tracers = :c)
                @test model isa NonhydrostaticModel

                model = NonhydrostaticModel(grid)
                @test model isa NonhydrostaticModel
            end
        end
    end

    @testset "Hydrostatic pressure anomaly with periodic vertical topology" begin
        @info "  Testing hydrostatic pressure anomaly with periodic vertical topology..."
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4), extent=(1, 1), topology=(Flat, Bounded, Periodic))
            model = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
            @test isnothing(model.pressures.pHY′)

            model = NonhydrostaticModel(grid; buoyancy=nothing)
            @test isnothing(model.pressures.pHY′)

            model = NonhydrostaticModel(grid; buoyancy = BuoyancyTracer(), tracers = :b)
            @test isnothing(model.pressures.pHY′)
        end
    end

    @testset "Setting model fields" begin
        @info "  Testing setting model fields..."
        for arch in archs, FT in float_types
            N = (4, 4, 4)
            L = (2π, 3π, 5π)

            rectilinear_grid = RectilinearGrid(arch, FT, size=N, extent=L,
                                               topology = (Periodic, Bounded, Bounded))
            latlon_grid = LatitudeLongitudeGrid(arch, FT; size=N, latitude=(-1, 1), longitude=(-1, 1), z=(-100, 0),
                                                topology = (Periodic, Bounded, Bounded))

            for grid in (rectilinear_grid, latlon_grid)
                model = NonhydrostaticModel(grid; buoyancy = SeawaterBuoyancy(), tracers = (:T, :S))

                u, v, w = model.velocities
                T, S = model.tracers

                # Test setting an array
                T₀_array = rand(FT, size(grid)...)
                T_answer = deepcopy(T₀_array)

                set!(model; enforce_incompressibility=false, T=T₀_array)

                @test Array(interior(T)) ≈ T_answer

                # Test setting functions
                u₀(x, y, z) = 1 + x + y + z
                v₀(x, y, z) = 2 + sin(x * y * z)
                w₀(x, y, z) = 3 + y * z
                T₀(x, y, z) = 4 + tanh(x + y - z)
                S₀(x, y, z) = 5

                set!(model, enforce_incompressibility=false, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

                xC, yC, zC = nodes(model.grid, (Center(), Center(), Center()), reshape=true)
                xF, yF, zF = nodes(model.grid, (Face(),   Face(),   Face()), reshape=true)

                # Form solution arrays
                u_answer = u₀.(xF, yC, zC) |> Array
                v_answer = v₀.(xC, yF, zC) |> Array
                w_answer = w₀.(xC, yC, zF) |> Array
                T_answer = T₀.(xC, yC, zC) |> Array
                S_answer = S₀.(xC, yC, zC) |> Array

                Nx, Ny, Nz = size(model.grid)

                cpu_grid = on_architecture(CPU(), grid)

                u_cpu = XFaceField(cpu_grid)
                v_cpu = YFaceField(cpu_grid)
                w_cpu = ZFaceField(cpu_grid)
                T_cpu = CenterField(cpu_grid)
                S_cpu = CenterField(cpu_grid)

                set!(u_cpu, u)
                set!(v_cpu, v)
                set!(w_cpu, w)
                set!(T_cpu, T)
                set!(S_cpu, S)

                values_match = [
                                all(u_answer .≈ interior(u_cpu)),
                                all(v_answer[:, 2:Ny, :] .≈ interior(v_cpu)[:, 2:Ny, :]),
                                all(w_answer[:, :, 2:Nz] .≈ interior(w_cpu)[:, :, 2:Nz]),
                                all(T_answer .≈ interior(T_cpu)),
                                all(S_answer .≈ interior(S_cpu)),
                               ]

                @test all(values_match)

                # Test whether set! copies boundary conditions
                # Note: we need to cleanup broadcasting for this -- see https://github.com/CliMA/Oceananigans.jl/pull/2786/files#r1008955571
                @test u_cpu[1, 1, 1] == u_cpu[Nx+1, 1, 1]  # x-periodicity
                @test all(u_cpu[1:Nx, 1:Ny, 1] .== u_cpu[1:Nx, 1:Ny, 0])     # free slip at bottom
                @test all(u_cpu[1:Nx, 1:Ny, Nz] .== u_cpu[1:Nx, 1:Ny, Nz+1]) # free slip at top

                # Test that enforce_incompressibility works
                set!(model, u=0, v=0, w=1, T=0, S=0)

                # Note: Before PR #5021, which introduces a volume inverse norm for the conjugate gradient convergence criteria,
                # ϵ = 10 * eps(FT), see https://github.com/CliMA/Oceananigans.jl/pull/5021.
                # We relax the tolerance in order to reduce the number of iterations needed for convergence.
                # This affects the divergence of the latitude-longitude grid, increasingly it slightly from previous.
                ϵ = sqrt(eps(FT))
                set!(w_cpu, w)
                @test all(abs.(interior(w_cpu)) .< ϵ)

                # Test setting the background_fields to a Field
                U_field = XFaceField(grid)
                U_field .= 1
                model = NonhydrostaticModel(grid; background_fields = (u=U_field,))
                @test model.background_fields.velocities.u isa Field

                U_field = CenterField(grid)
                @test_throws ArgumentError NonhydrostaticModel(grid; background_fields = (u=U_field,))
            end
        end
    end

    @testset "Pressure solver dispatch" begin
        @info "  Testing pressure solver dispatch..."

        for arch in archs
            grid_xyz = RectilinearGrid(arch, size=(2,2,2), extent=(1,1,1))
            grid_xy  = RectilinearGrid(arch, size=(2,2,2), x=(0,1), y=(0,1),
                                       z=[0.0, 0.5, 1.0], topology=(Periodic, Periodic, Bounded))
            grid_xz  = RectilinearGrid(arch, size=(2,2,2), x=(0,1), z=(0,1),
                                       y=[-0.5, 0.0, 0.5], topology=(Periodic, Bounded, Bounded))
            grid_yz  = RectilinearGrid(arch, size=(2,2,2), y=(0,1), z=(0,1),
                                       x=[-0.5, 0.0, 0.5], topology=(Bounded, Periodic, Bounded))
            ibg_xyz  = ImmersedBoundaryGrid(grid_xyz, GridFittedBottom((x,y) -> 0.2))
            ibg_xz   = ImmersedBoundaryGrid(grid_xz,  GridFittedBottom((y,z) -> 0.2))

            # Rigid-lid (::Nothing) regressions
            @test nonhydrostatic_pressure_solver(arch, grid_xyz, nothing) isa FFTBasedPoissonSolver
            @test nonhydrostatic_pressure_solver(arch, grid_xy,  nothing) isa FourierTridiagonalPoissonSolver
            @test nonhydrostatic_pressure_solver(arch, grid_xz,  nothing) isa FourierTridiagonalPoissonSolver
            @test nonhydrostatic_pressure_solver(arch, grid_yz,  nothing) isa FourierTridiagonalPoissonSolver

            # Free-surface dispatch uses a mock object (only .gravitational_acceleration is needed for construction)
            mock_fs = (; gravitational_acceleration=9.81, displacement=nothing)

            # XYZReg + fs → FT with InhomogeneousFormulation (direct solve)
            @test nonhydrostatic_pressure_solver(arch, grid_xyz, mock_fs) isa FourierTridiagonalPoissonSolver

            # XYReg (stretched z) + fs → FT with InhomogeneousFormulation (z-tridiagonal still valid)
            @test nonhydrostatic_pressure_solver(arch, grid_xy, mock_fs) isa FourierTridiagonalPoissonSolver

            # XZReg (stretched y) + fs → CG with FreeSurfaceLaplacian and deflated FT preconditioner
            let solver = nonhydrostatic_pressure_solver(arch, grid_xz, mock_fs)
                @test solver isa ConjugateGradientPoissonSolver
                @test solver.conjugate_gradient_solver.linear_operation! isa FreeSurfaceLaplacian
                @test solver.conjugate_gradient_solver.preconditioner isa DeflatedFourierTridiagonalPreconditioner
            end

            # YZReg (stretched x) + fs → CG with FreeSurfaceLaplacian and deflated FT preconditioner
            let solver = nonhydrostatic_pressure_solver(arch, grid_yz, mock_fs)
                @test solver isa ConjugateGradientPoissonSolver
                @test solver.conjugate_gradient_solver.linear_operation! isa FreeSurfaceLaplacian
                @test solver.conjugate_gradient_solver.preconditioner isa DeflatedFourierTridiagonalPreconditioner
            end

            # IBG on XYZReg + fs → CG with FreeSurfaceLaplacian (FT InhomogZDir preconditioner)
            let solver = nonhydrostatic_pressure_solver(arch, ibg_xyz, mock_fs)
                @test solver isa ConjugateGradientPoissonSolver
                @test solver.conjugate_gradient_solver.linear_operation! isa FreeSurfaceLaplacian
                @test solver.conjugate_gradient_solver.preconditioner isa FourierTridiagonalPoissonSolver
            end

            # IBG on XZReg + fs → CG with FreeSurfaceLaplacian and deflated FT preconditioner
            let solver = nonhydrostatic_pressure_solver(arch, ibg_xz, mock_fs)
                @test solver isa ConjugateGradientPoissonSolver
                @test solver.conjugate_gradient_solver.linear_operation! isa FreeSurfaceLaplacian
                @test solver.conjugate_gradient_solver.preconditioner isa DeflatedFourierTridiagonalPreconditioner
            end
        end
    end

    @testset "NonhydrostaticModel with implicit free surface" begin
        @info "  Testing NonhydrostaticModel with implicit free surface..."

        for arch in archs
            # XYZReg + ImplicitFreeSurface: FT direct solve (existing behavior, regression test)
            grid = RectilinearGrid(arch, size=(2,2,2), extent=(1,1,1),
                                   topology=(Periodic, Periodic, Bounded))
            model = NonhydrostaticModel(grid; free_surface=ImplicitFreeSurface())
            @test model.pressure_solver isa FourierTridiagonalPoissonSolver
            @test !isnothing(model.free_surface)

            # IBG on XYZReg + ImplicitFreeSurface: CG with FreeSurfaceLaplacian (new)
            ibg = ImmersedBoundaryGrid(grid, GridFittedBottom((x,y) -> 0.2))
            ibg_model = NonhydrostaticModel(ibg; free_surface=ImplicitFreeSurface())
            @test ibg_model.pressure_solver isa ConjugateGradientPoissonSolver
            @test ibg_model.pressure_solver.conjugate_gradient_solver.linear_operation! isa FreeSurfaceLaplacian
            @test !isnothing(ibg_model.free_surface)

            # Stretched-x grid + ImplicitFreeSurface: constructible (the free surface is materialized
            # without the hydrostatic implicit step solver) and steppable
            stretched_grid = RectilinearGrid(arch, size=(2,2,2), y=(0,1), z=(0,1),
                                             x=[0.0, 0.4, 1.0], topology=(Bounded, Periodic, Bounded))
            stretched_model = NonhydrostaticModel(stretched_grid; free_surface=ImplicitFreeSurface())
            @test stretched_model.pressure_solver isa ConjugateGradientPoissonSolver
            time_step!(stretched_model, 0.01)
            @test iteration(stretched_model.pressure_solver) < stretched_model.pressure_solver.conjugate_gradient_solver.maxiter
        end
    end

    @testset "Free surface solution: CG solver vs direct FT solve" begin
        @info "  Testing free surface solution agreement between CG and FT solvers..."

        # Step a free-surface model and return (η, u, w)
        function stepped_free_surface_fields(model; Δt=0.01, N=10)
            set!(model.free_surface.displacement, (x, z) -> 0.05 * cospi(x))
            for _ in 1:N
                time_step!(model, Δt)
            end
            return (Array(interior(model.free_surface.displacement)),
                    Array(interior(model.velocities.u)),
                    Array(interior(model.velocities.w)))
        end

        for arch in archs
            Nx, Nz = 16, 8
            free_surface = ImplicitFreeSurface()
            grid = RectilinearGrid(arch, size=(Nx, Nz), x=(0, 1), z=(-1, 0),
                                   topology=(Bounded, Flat, Bounded))
            η_ft, u_ft, w_ft = stepped_free_surface_fields(NonhydrostaticModel(grid; free_surface))

            # CG + FreeSurfaceLaplacian on the same grid reproduces the direct FT solve
            pressure_solver = ConjugateGradientPoissonSolver(grid;
                                                             linear_operation = FreeSurfaceLaplacian(),
                                                             preconditioner = fft_free_surface_preconditioner(grid),
                                                             enforce_gauge_condition! = no_gauge_enforcement!)
            η_cg, u_cg, w_cg = stepped_free_surface_fields(NonhydrostaticModel(grid; free_surface, pressure_solver))
            @test isapprox(η_cg, η_ft, atol=1e-8)
            @test isapprox(u_cg, u_ft, atol=1e-8)
            @test isapprox(w_cg, w_ft, atol=1e-8)

            # An x-spacing array makes the grid stretched-typed (YZRegularRG), dispatching to
            # CG + deflated FT preconditioner; with uniform values the FT solution is the reference.
            array_x_grid = RectilinearGrid(arch, size=(Nx, Nz), x=collect(range(0, 1, length=Nx+1)),
                                           z=(-1, 0), topology=(Bounded, Flat, Bounded))
            array_x_model = NonhydrostaticModel(array_x_grid; free_surface)
            @test array_x_model.pressure_solver isa ConjugateGradientPoissonSolver
            η_sx, u_sx, w_sx = stepped_free_surface_fields(array_x_model)
            @test isapprox(η_sx, η_ft, atol=1e-8)
            @test isapprox(u_sx, u_ft, atol=1e-8)
            @test isapprox(w_sx, w_ft, atol=1e-8)

            # IBG with a flat bottom on a face of the underlying grid reproduces the
            # FT solution on the equivalent smaller grid
            underlying = RectilinearGrid(arch, size=(Nx, 2Nz), x=(0, 1), z=(-2, 0),
                                         topology=(Bounded, Flat, Bounded))
            ibg = ImmersedBoundaryGrid(underlying, GridFittedBottom(x -> -1))
            ibg_model = NonhydrostaticModel(ibg; free_surface)
            @test ibg_model.pressure_solver isa ConjugateGradientPoissonSolver
            η_ib, u_ib, w_ib = stepped_free_surface_fields(ibg_model)
            @test isapprox(η_ib[:, :, 1], η_ft[:, :, 1], atol=1e-8)
            @test isapprox(u_ib[:, :, Nz+1:2Nz], u_ft, atol=1e-8)
            @test isapprox(w_ib[:, :, Nz+1:2Nz+1], w_ft, atol=1e-8)
        end
    end
end
