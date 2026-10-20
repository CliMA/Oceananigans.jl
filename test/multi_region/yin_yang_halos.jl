include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.Grids: with_halo
using Oceananigans.MultiRegion: YinYangInterpolationTable, yin_yang_interpolation, interpolation_table,
                                stencil_start, lagrange_weights

const vector_locations = (((Face(), Center()), (Center(), Face())), ((Face(), Face()), (Face(), Face())))

# Geographic velocity of a wavenumber-4 Rossby-Haurwitz wave
rossby_haurwitz_velocity(λ, φ; R = 4) = (cosd(φ) + cosd(φ)^(R-1) * (R * sind(φ)^2 - cosd(φ)^2) * cosd(R * λ),
                                         -R * cosd(φ)^(R-1) * sind(φ) * sind(R * λ))

function scalar_halo_error(grid, ℓx, ℓy, f = spherical_harmonics)
    c = location_field(ℓx, ℓy, grid)
    set!(c, (λ, φ, z) -> f(λ, φ))
    fill_halo_regions!(c)
    return panel_error(c, (λ, φ, r) -> f(λ, φ))
end

function vector_halo_error(grid, velocity, (ℓu, ℓv); fill_vector_halos! = (u, v) -> fill_halo_regions!((u, v)))
    u = location_field(ℓu..., grid)
    v = location_field(ℓv..., grid)
    set_panel_components!(u, v, velocity)
    fill_vector_halos!(u, v)
    return max(panel_error(u, panel_u(velocity, grid)), panel_error(v, panel_v(velocity, grid)))
end

@testset "Yin-Yang halo filling [$(typeof(arch))]" for arch in archs
    grid = YinYangGrid(arch; size=(48, 16, 2), z=(-1, 0))
    Nx, Ny, _ = size(grid)

    @testset "Interpolation tables [$FT]" for FT in float_types
        typed_grid = YinYangGrid(arch, FT; size=(48, 16, 1), z=(-1, 0))
        interpolation = yin_yang_interpolation(typed_grid)
        receivers = halo_nodes(typed_grid)
        @test summary(interpolation) ==
            "YinYangInterpolation with $(length(receivers)) receivers per location and 4×4 stencils"

        for name in propertynames(interpolation)
            table = on_architecture(CPU(), getproperty(interpolation, name))
            @test table isa YinYangInterpolationTable{4}
            @test eltype(table.weights.x) == FT
            @test collect(zip(table.receivers.i, table.receivers.j)) == receivers
            @test all(1 .≤ table.donors.i .≤ Nx - 3)
            @test all(1 .≤ table.donors.j .≤ Ny - 3)
            @test sum(table.weights.x, dims=2) ≈ ones(FT, length(receivers), 1)
            @test sum(table.weights.y, dims=2) ≈ ones(FT, length(receivers), 1)
            @test table.rotation.cosψ.^2 .+ table.rotation.sinψ.^2 ≈ ones(FT, length(receivers))
        end

        @test interpolation_table(interpolation, Center(), Center()) === interpolation.center_center
        @test interpolation_table(interpolation, Face(), Center(), Center(), Face()) ===
              interpolation.face_center_from_center_face
        @test yin_yang_interpolation(with_halo((4, 4, 4), typed_grid)).center_center isa YinYangInterpolationTable{4}
    end

    @testset "The tables serve both panels" begin
        interpolation = yin_yang_interpolation(grid)
        panels = (on_architecture(CPU(), grid[1]), on_architecture(CPU(), grid[2]))
        for (ℓʳ, ℓᵈ, name) in (((Center(), Center()), (Center(), Center()), :center_center),
                               ((Face(), Face()), (Face(), Face()), :face_face),
                               ((Face(), Center()), (Center(), Face()), :face_center_from_center_face)),
            (receiver, donor) in ((1, 2), (2, 1))

            table = on_architecture(CPU(), getproperty(interpolation, name))
            receivers = zip(table.receivers.i, table.receivers.j)
            λφ = [node_coordinates(i, j, panels[receiver], ℓʳ...) for (i, j) in receivers]
            indices = [geographic_fractional_indices(λ, φ, grid, donor, ℓᵈ...) for (λ, φ) in λφ]
            i★, j★ = first.(indices), last.(indices)
            poles = north_pole(panels[receiver]), north_pole(panels[donor])
            ψ = [rotated_basis_angle(λ, φ, poles[1]...) - rotated_basis_angle(λ, φ, poles[2]...) for (λ, φ) in λφ]
            weights(x★, x₀) = permutedims(reduce(hcat, lagrange_weights.(x★, x₀, 4)))

            # The donor stencil is one of the two brackets of the fractional index
            @test maximum(abs.(stencil_start.(i★, 4) .- table.donors.i)) ≤ 1
            @test maximum(abs.(stencil_start.(j★, 4) .- table.donors.j)) ≤ 1
            @test table.weights.x ≈ weights(i★, table.donors.i) rtol=1e-6
            @test table.weights.y ≈ weights(j★, table.donors.j) rtol=1e-6
            @test table.rotation.cosψ ≈ cos.(ψ) rtol=1e-10
            @test table.rotation.sinψ ≈ sin.(ψ) rtol=1e-10
        end
    end

    @testset "Scalar halo fills" begin
        errors = [scalar_halo_error(YinYangGrid(arch; size=(3Nφ, Nφ, 1), z=(-1, 0)), Center(), Center())
                  for Nφ in (16, 32)]
        @test errors[1] < 1e-3
        @test only(convergence_rates(errors)) > 3.5
        @test all(scalar_halo_error(grid, ℓx, ℓy) < 1e-3 for (ℓx, ℓy) in yin_yang_locations)
        @test scalar_halo_error(YinYangGrid(arch, Float32; size=(48, 16, 1), z=(-1, 0)), Face(), Center()) < 1e-3

        bilinear = YinYangGrid(arch; size=(48, 16, 1), z=(-1, 0), stencil_width=2)
        @test yin_yang_interpolation(bilinear).center_center isa YinYangInterpolationTable{2}
        @test errors[1] < scalar_halo_error(bilinear, Center(), Center()) < 1e-1

        # The fill reaches the vertical halos, and ignores the extra arguments and keywords
        c = CenterField(grid)
        set!(c, (λ, φ, z) -> spherical_harmonics(λ, φ))
        fill_halo_regions!(c, nothing, nothing; async=true, only_local_halos=true)
        for r in 1:2
            data = on_architecture(CPU(), getregion(c, r))
            @test data[1, 1, 0] == data[1, 1, 1]
            @test data[-2, 25, 3] == data[-2, 25, 2]
        end
        @test c.communication_buffers === nothing

        # Fields windowed in x or y get only their vertical halos
        for indices in ((1:10, :, :), (:, 1:5, :), (:, :, 2:2))
            d = CenterField(grid; indices)
            set!(d, 1)
            fill_halo_regions!(d)
            @test all(==(1), interior(getregion(d, 2)))
        end
        set!(c, 2)
        c² = Field(c * c; indices=(1:10, :, :))
        compute!(c²)
        @test all(==(4), interior(getregion(c², 1)))
    end

    @testset "Vector halo fills" begin
        for velocity in (solid_body_velocity, rossby_haurwitz_velocity), locations in vector_locations
            @test vector_halo_error(grid, velocity, locations; fill_vector_halos! = fill_halo_regions!) < 2e-3
        end

        # Only the pair (u, v) at (Face, Center) and (Center, Face) is a vector; two (Face, Face) fields in a tuple,
        # and the components in a longer tuple, are filled as scalars, which is wrong for a vector
        @test vector_halo_error(grid, solid_body_velocity, vector_locations[1]) < 1e-5
        @test vector_halo_error(grid, solid_body_velocity, vector_locations[2]) > 0.1
        fill_with_tracer!(u, v) = fill_halo_regions!((u, v, CenterField(grid)))
        @test vector_halo_error(grid, solid_body_velocity, vector_locations[1]; fill_vector_halos! = fill_with_tracer!) > 0.1

        # With signed = false the components are filled as scalars
        u, v = XFaceField(grid), YFaceField(grid)
        set_panel_components!(u, v, solid_body_velocity)
        fill_halo_regions!((u, v); signed=false)
        unsigned_parents = (regional_parents(u), regional_parents(v))
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        @test (regional_parents(u), regional_parents(v)) == unsigned_parents

        # Fields without boundary conditions have no halos to fill
        u, v = XFaceField(grid; boundary_conditions=nothing), YFaceField(grid; boundary_conditions=nothing)
        @test fill_halo_regions!((u, v)) === nothing
    end
end
