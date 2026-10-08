include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Fields: VelocityFields
using Oceananigans.BoundaryConditions: UPivot, TPivot, FPivot, regularize_field_boundary_conditions
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: calculate_substeps,
                                                                                  calculate_adaptive_settings,
                                                                                  constant_averaging_kernel,
                                                                                  materialize_free_surface,
                                                                                  SplitExplicitFreeSurface,
                                                                                  iterate_split_explicit!,
                                                                                  weights_from_substeps
using Oceananigans.Architectures: convert_to_device
using Oceananigans.Utils: capture_launches, StepValue

@inline noforcing(args...) = 0

barotropic_boundary_conditions(grid) =
    (U = FieldBoundaryConditions(grid, (Face(), Center(), nothing)),
     V = FieldBoundaryConditions(grid, (Center(), Face(), nothing)))

clock = Clock{Float64}(time=0)

@testset "Split-Explicit Dynamics" begin

    for FT in float_types
        for arch in archs
            topology = (Periodic, Periodic, Bounded)

            Nx, Ny, Nz = 128, 64, 1
            Lx = Ly = 2π
            Lz = 1 / Oceananigans.defaults.gravitational_acceleration

            grid = RectilinearGrid(arch, FT;
                                   topology, size = (Nx, Ny, Nz),
                                   x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                                   halo = (1, 1, 1))

            velocities = VelocityFields(grid)

            sefs = SplitExplicitFreeSurface(substeps = 200, averaging_kernel = constant_averaging_kernel)
            sefs = materialize_free_surface(sefs, velocities, grid, barotropic_boundary_conditions(grid))

            sefs.displacement .= 0
            GU = Field{Face, Center, Nothing}(grid)
            GV = Field{Center, Face, Nothing}(grid)

            @testset " One timestep test " begin
                state = sefs.filtered_state
                U, V  = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅

                η = sefs.displacement
                Δτ = 1.0

                η₀(x, y, z) = sin(x)
                set!(η, η₀)

                Nsubsteps = calculate_substeps(sefs.substepping, 1)
                fractional_Δt, weights, transport_weights = calculate_adaptive_settings(sefs.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights

                iterate_split_explicit!(sefs, grid, GU, GV, Δτ, noforcing, clock, weights, transport_weights, Val(1))

                U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
                U_exact = (reshape(-cos.(grid.xᶠᵃᵃ), (length(grid.xᶜᵃᵃ), 1)).+reshape(0 * grid.yᵃᶜᵃ, (1, length(grid.yᵃᶜᵃ))))[2:Nx+1, 2:Ny+1]

                @test maximum(abs.(U_exact - U_computed)) < 1e-3
            end

            @testset "Multi-timestep test " begin
                state = sefs.filtered_state
                U, V = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
                η = sefs.displacement

                T  = 2π
                Δτ = 2π / maximum([Nx, Ny]) * 5e-2 # the last factor is essentially the order of accuracy
                Nt = floor(Int, T / Δτ)
                Δτ_end = T - Nt * Δτ

                sefs = SplitExplicitFreeSurface(substeps = Nt, averaging_kernel = constant_averaging_kernel)
                sefs = materialize_free_surface(sefs, velocities, grid, barotropic_boundary_conditions(grid))

                # set!(η, f(x, y))
                η₀(x, y, z) = sin(x)
                set!(η, η₀)
                set!(U, 0)
                set!(V, 0)

                η̅  .= 0
                U̅  .= 0
                V̅  .= 0
                GU .= 0
                GV .= 0

                weights = sefs.substepping.averaging_weights

                for _ in 1:Nt
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, noforcing, clock, weights, weights, Val(1))
                end
                iterate_split_explicit!(sefs, grid, GU, GV, Δτ, noforcing, clock, weights, weights, Val(1))

                U_computed = Array(deepcopy(interior(U)))
                η_computed = Array(deepcopy(interior(η)))
                set!(η, η₀)
                set!(U, 0)
                U_exact = Array(deepcopy(interior(U)))
                η_exact = Array(deepcopy(interior(η)))

                @test maximum(abs.(U_computed - U_exact)) < 1e-3
                @test maximum(abs.(η_computed - η_exact)) < max(100eps(FT), 1e-6)
            end

            sefs = SplitExplicitFreeSurface(substeps = 200, averaging_kernel = constant_averaging_kernel)
            sefs = materialize_free_surface(sefs, velocities, grid, barotropic_boundary_conditions(grid))

            sefs.displacement .= 0

            @testset "Averaging / Do Nothing test " begin
                state = sefs.filtered_state
                U, V  = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
                η = sefs.displacement
                g = sefs.gravitational_acceleration

                Δτ = 2π / maximum([Nx, Ny]) * 1e-2 # the last factor is essentially the order of accuracy

                # set!(η, f(x, y))
                η_avg = 1
                U_avg = 2
                V_avg = 3
                fill!(η, η_avg)
                fill!(U, U_avg)
                fill!(V, V_avg)

                fill!(η̅ , 0)
                fill!(U̅ , 0)
                fill!(V̅ , 0)
                fill!(GU, 0)
                fill!(GV, 0)

                Nsubsteps  = calculate_substeps(sefs.substepping, 1)
                fractional_Δt, weights, transport_weights = calculate_adaptive_settings(sefs.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights

                for step in 1:Nsubsteps
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, noforcing, clock, weights, transport_weights, Val(1))
                end

                U_computed = Array(deepcopy(interior(U)))
                V_computed = Array(deepcopy(interior(V)))
                η_computed = Array(deepcopy(interior(η)))

                U̅_computed = Array(deepcopy(interior(U̅)))
                V̅_computed = Array(deepcopy(interior(V̅)))
                η̅_computed = Array(deepcopy(interior(η̅)))

                tolerance = 100eps(FT)

                @test maximum(abs.(U_computed .- U_avg)) < tolerance
                @test maximum(abs.(η_computed .- η_avg)) < tolerance
                @test maximum(abs.(V_computed .- V_avg)) < tolerance

                @test maximum(abs.(U̅_computed .- U_avg)) < tolerance
                @test maximum(abs.(η̅_computed .- η_avg)) < tolerance
                @test maximum(abs.(V̅_computed .- V_avg)) < tolerance
            end

            @testset "Complex Multi-Timestep " begin
                # Test 3: Testing analytic solution to
                # ∂ₜη + ∇⋅U̅ = 0
                # ∂ₜU̅ + ∇η  = G̅
                kx = 2
                ky = 3
                ω = sqrt(kx^2 + ky^2)
                T = 2π / ω / 3 * 2
                Δτ = 2π / maximum([Nx, Ny]) * 1e-2 # error mostly spatially dependent, except in the averaging
                Nt = floor(Int, T / Δτ)
                Δτ_end = T - Nt * Δτ

                sefs = SplitExplicitFreeSurface(grid; substeps = Nt + 1, averaging_kernel = constant_averaging_kernel)
                sefs = materialize_free_surface(sefs, velocities, grid, barotropic_boundary_conditions(grid))

                state = sefs.filtered_state
                U, V = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
                η = sefs.displacement
                g = sefs.gravitational_acceleration

                # set!(η, f(x, y)) k² = ω²
                gu_c = 1
                gv_c = 2
                η₀(x, y, z) = sin(kx * x) * sin(ky * y) + 1
                set!(η, η₀)

                η_mean_before = mean(Array(interior(η)))

                U .= 0 # so that ∂ₜη(t=0) = 0
                V .= 0 # so that ∂ₜη(t=0) = 0
                η̅ .= 0
                U̅ .= 0
                V̅ .= 0
                GU .= gu_c
                GV .= gv_c

                weights = sefs.substepping.averaging_weights
                for i in 1:Nt
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, noforcing, clock, weights, weights, Val(1))
                end
                iterate_split_explicit!(sefs, grid, GU, GV, Δτ, noforcing, clock, weights, weights, Val(1))

                η_mean_after = mean(Array(interior(η)))

                tolerance = 10eps(FT)
                @test abs(η_mean_after - η_mean_before) < tolerance

                η_computed = Array(deepcopy(interior(η, :, 1, 1)))
                U_computed = Array(deepcopy(interior(U, :, 1, 1)))
                V_computed = Array(deepcopy(interior(V, :, 1, 1)))

                η̅_computed = Array(deepcopy(interior(η̅, :, 1, 1)))
                U̅_computed = Array(deepcopy(interior(U̅, :, 1, 1)))
                V̅_computed = Array(deepcopy(interior(V̅, :, 1, 1)))

                set!(η, η₀)

                # ∂ₜₜ(η) = Δη
                η_exact = cos(ω * T) * (Array(interior(η, :, 1, 1)) .- 1) .+ 1

                U₀(x, y) = kx * cos(kx * x) * sin(ky * y) # ∂ₜU = - ∂x(η), since we know η
                set!(U, U₀)
                U_exact = -(sin(ω * T) * 1 / ω) .* Array(interior(U, :, 1, 1)) .+ gu_c * T

                V₀(x, y) = ky * sin(kx * x) * cos(ky * y) # ∂ₜV = - ∂y(η), since we know η
                set!(V, V₀)
                V_exact = -(sin(ω * T) * 1 / ω) .* Array(interior(V, :, 1, 1)) .+ gv_c * T

                η̅_exact = (sin(ω * T) / ω - sin(ω * 0) / ω) / T * (Array(interior(η, :, 1, 1)) .- 1) .+ 1
                U̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(interior(U, :, 1, 1)) .+ gu_c * T / 2
                V̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(interior(V, :, 1, 1)) .+ gv_c * T / 2

                tolerance = 1e-2

                @test maximum(abs.(U_computed - U_exact)) / maximum(abs.(U_exact)) < tolerance
                @test maximum(abs.(V_computed - V_exact)) / maximum(abs.(V_exact)) < tolerance
                @test maximum(abs.(η_computed - η_exact)) / maximum(abs.(η_exact)) < tolerance

                @test maximum(abs.(U̅_computed - U̅_exact)) < tolerance
                @test maximum(abs.(V̅_computed - V̅_exact)) < tolerance
                @test maximum(abs.(η̅_computed - η̅_exact)) < tolerance
            end
        end # end of architecture loop
    end # end of float type loop
end # end of testset loop

@testset "extend_halos vs fill_halos consistency" begin
    for arch in archs
        topology = (Periodic, Periodic, Bounded)
        Nx, Ny, Nz = 32, 32, 1
        Lx = Ly = 2π
        Lz = 1 / Oceananigans.defaults.gravitational_acceleration

        grid = RectilinearGrid(arch, Float64;
                               topology, size = (Nx, Ny, Nz),
                               x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                               halo = (1, 1, 1))

        velocities = VelocityFields(grid)
        Nsubsteps = 30

        # Create two free surfaces: one with extended halos, one that fills halos each substep
        sefs_extend = SplitExplicitFreeSurface(grid; substeps = Nsubsteps,
                                               averaging_kernel = constant_averaging_kernel,
                                               extend_halos = true)
        sefs_extend = materialize_free_surface(sefs_extend, velocities, grid, barotropic_boundary_conditions(grid))

        sefs_fill = SplitExplicitFreeSurface(grid; substeps = Nsubsteps,
                                             averaging_kernel = constant_averaging_kernel,
                                             extend_halos = false)
        sefs_fill = materialize_free_surface(sefs_fill, velocities, grid, barotropic_boundary_conditions(grid))

        # Slow barotropic forcing
        GU = Field{Face, Center, Nothing}(grid)
        GV = Field{Center, Face, Nothing}(grid)
        GU .= 0
        GV .= 0

        # Initial condition
        η₀(x, y, z) = sin(x) * cos(y)

        for (label, sefs) in [("extend_halos", sefs_extend), ("fill_halos", sefs_fill)]
            set!(sefs.displacement, η₀)
            sefs.barotropic_velocities.U .= 0
            sefs.barotropic_velocities.V .= 0
            for field in sefs.filtered_state
                fill!(field, 0)
            end
        end

        Δτ = 1.0
        fractional_Δt, weights, transport_weights = calculate_adaptive_settings(sefs_extend.substepping, Nsubsteps)

        iterate_split_explicit!(sefs_extend, sefs_extend.displacement.grid, GU, GV, Δτ, noforcing, clock, weights, transport_weights, Val(Nsubsteps))

        fractional_Δt, weights, transport_weights = calculate_adaptive_settings(sefs_fill.substepping, Nsubsteps)

        iterate_split_explicit!(sefs_fill, grid, GU, GV, Δτ, noforcing, clock, weights, transport_weights, Val(Nsubsteps))

        # Compare: both should give the same interior result
        η_extend = Array(interior(sefs_extend.displacement))
        η_fill   = Array(interior(sefs_fill.displacement))
        U_extend = Array(interior(sefs_extend.barotropic_velocities.U))
        U_fill   = Array(interior(sefs_fill.barotropic_velocities.U))
        V_extend = Array(interior(sefs_extend.barotropic_velocities.V))
        V_fill   = Array(interior(sefs_fill.barotropic_velocities.V))

        @test η_extend ≈ η_fill
        @test U_extend ≈ U_fill
        @test V_extend ≈ V_fill

        η̅_extend = Array(interior(sefs_extend.filtered_state.η̅))
        η̅_fill   = Array(interior(sefs_fill.filtered_state.η̅))
        U̅_extend = Array(interior(sefs_extend.filtered_state.U̅))
        U̅_fill   = Array(interior(sefs_fill.filtered_state.U̅))

        @test η̅_extend ≈ η̅_fill
        @test U̅_extend ≈ U̅_fill
    end
end

@inline clock_dependent_forcing(i, j, k, grid, clock, fields) = 1e-3 * sin(clock.time)

@testset "Device-backed substep values and replayed barotropic graphs" begin
    for arch in archs
        grid = RectilinearGrid(arch; size = (16, 16, 1), x = (0, 2π), y = (0, 2π), z = (-1, 0),
                               topology = (Periodic, Periodic, Bounded))

        free_surface = SplitExplicitFreeSurface(grid; substeps = 10)
        free_surface = materialize_free_surface(free_surface, VelocityFields(grid), grid, barotropic_boundary_conditions(grid))

        GU = Field{Face, Center, Nothing}(grid)
        GV = Field{Center, Face, Nothing}(grid)
        set!(GU, (x, y) -> 1e-4 * cos(y))
        set!(GV, (x, y) -> 1e-4 * sin(x))

        η = free_surface.displacement
        U, V = free_surface.barotropic_velocities
        state = free_surface.filtered_state
        barotropic_fields = (η, U, V, state.η̅, state.U̅, state.V̅, state.Ũ, state.Ṽ)

        function substep_settings(substeps)
            fractional_Δt, weights, transport_weights = weights_from_substeps(eltype(grid), substeps, constant_averaging_kernel)
            return fractional_Δt * 10, weights, transport_weights
        end

        # Substep the barotropic mode once, from the same initial state every time, and return the result
        function substep_barotropic_mode(Δτ, clock, weights, transport_weights; capture_graphs, GU = GU)
            foreach(field -> fill!(field, 0), barotropic_fields)
            set!(η, (x, y, z) -> 1e-2 * sin(x) * cos(y))
            fill_halo_regions!(η)

            capture_launches[] = capture_graphs
            try
                iterate_split_explicit!(free_surface, grid, GU, GV, Δτ, clock_dependent_forcing, clock,
                                        weights, transport_weights, Val(length(weights)))
            finally
                capture_launches[] = true
            end

            return map(field -> Array(interior(field)), barotropic_fields)
        end

        Δτ₁, weights₁, transport_weights₁ = substep_settings(10)
        Δτ₂, weights₂, transport_weights₂ = substep_settings(12)
        clock₁ = Clock{Float64}(time = 1)
        clock₂ = Clock{Float64}(time = 2)

        # References, computed by launching the substepping kernels one by one
        reference₁     = substep_barotropic_mode(Δτ₁, clock₁, weights₁, transport_weights₁; capture_graphs = false)
        clocked_ref₁   = substep_barotropic_mode(Δτ₁, clock₂, weights₁, transport_weights₁; capture_graphs = false)
        reference₂     = substep_barotropic_mode(Δτ₂, clock₁, weights₂, transport_weights₂; capture_graphs = false)

        @test any(!iszero, reference₁[1])
        @test reference₁ != clocked_ref₁ # the forcing depends on the clock, so the clock has to reach the kernel
        @test reference₁ != reference₂

        # Substep values read from device memory
        step = on_architecture(arch, [(; Δτ = Δτ₁, clock = convert_to_device(arch, clock₁))])
        device_values = substep_barotropic_mode(StepValue{:Δτ}(step), StepValue{:clock}(step),
                                                weights₁, transport_weights₁; capture_graphs = false)
        @test device_values == reference₁

        # The same substepping, recorded in and replayed from a CUDA graph
        @test substep_barotropic_mode(Δτ₁, clock₁, weights₁, transport_weights₁; capture_graphs = true) == reference₁   # capture
        @test substep_barotropic_mode(Δτ₁, clock₂, weights₁, transport_weights₁; capture_graphs = true) == clocked_ref₁ # replay with a new clock
        @test substep_barotropic_mode(Δτ₂, clock₁, weights₂, transport_weights₂; capture_graphs = true) == reference₂   # new weights, new graph
        @test substep_barotropic_mode(Δτ₁, clock₁, weights₁, transport_weights₁; capture_graphs = true) == reference₁   # the first graph is still cached

        other_GU = Field{Face, Center, Nothing}(grid)
        set!(other_GU, (x, y) -> 2e-4 * sin(y))
        other_reference = substep_barotropic_mode(Δτ₁, clock₁, weights₁, transport_weights₁; capture_graphs = false, GU = other_GU)
        @test substep_barotropic_mode(Δτ₁, clock₁, weights₁, transport_weights₁; capture_graphs = true, GU = other_GU) == other_reference
    end
end

@testset "extend_halos vs fill_halos consistency on tripolar grids" begin
    # Land over the two singular poles of the fold, at (70°E, 55°N) and (250°E, 55°N)
    cosine_of_pole_distance(λ, φ, pole_longitude) = sind(φ) * sind(55) + cosd(φ) * cosd(55) * cosd(λ - pole_longitude)
    Lz = 1 / Oceananigans.defaults.gravitational_acceleration
    bottom_height(λ, φ) = max(cosine_of_pole_distance(λ, φ, 70), cosine_of_pole_distance(λ, φ, 250)) > cosd(3) ? 0 : -Lz

    for arch in archs, (fold_topology, pivot) in ((RightCenterFolded, UPivot), (RightCenterFolded, TPivot), (RightFaceFolded, FPivot))
        underlying_grid = TripolarGrid(arch; size = (16, 10, 1), z = (-Lz, 0), north_poles_latitude = 55, first_pole_longitude = 70,
                                       fold_topology, pivot)
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

        velocities = VelocityFields(grid)
        boundary_conditions = (U = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :U),
                               V = regularize_field_boundary_conditions(FieldBoundaryConditions(), grid, :V))

        GU = Field{Face, Center, Nothing}(grid)
        GV = Field{Center, Face, Nothing}(grid)

        Nsubsteps = 30
        Δτ = 1e4

        extended, filled = map((true, false)) do extend_halos
            free_surface = SplitExplicitFreeSurface(grid; substeps = Nsubsteps, extend_halos, averaging_kernel = constant_averaging_kernel)
            free_surface = materialize_free_surface(free_surface, velocities, grid, boundary_conditions)
            set!(free_surface.displacement, (λ, φ, z) -> cosd(φ) * sind(λ - 20) + cosd(φ)^3 * cosd(3λ))

            fractional_Δt, weights, transport_weights = calculate_adaptive_settings(free_surface.substepping, Nsubsteps)
            iterate_split_explicit!(free_surface, grid, GU, GV, Δτ, noforcing, clock, weights, transport_weights, Val(Nsubsteps))

            return free_surface
        end

        # U is not compared: halo filling leaves 0/0 at the land-masked U-pivot poles
        @test Array(interior(extended.displacement)) ≈ Array(interior(filled.displacement))
        @test Array(interior(extended.barotropic_velocities.V)) ≈ Array(interior(filled.barotropic_velocities.V))
        @test Array(interior(extended.filtered_state.η̅)) ≈ Array(interior(filled.filtered_state.η̅))
    end
end
