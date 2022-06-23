include("dependencies_for_runtests.jl")

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Advection: 
        _symmetric_interpolate_xᶠᵃᵃ,
        _symmetric_interpolate_xᶜᵃᵃ,
        _symmetric_interpolate_yᵃᶠᵃ,
        _symmetric_interpolate_yᵃᶜᵃ,
        _left_biased_interpolate_xᶜᵃᵃ, 
        _left_biased_interpolate_xᶠᵃᵃ, 
        _right_biased_interpolate_xᶜᵃᵃ,
        _right_biased_interpolate_xᶠᵃᵃ,
        _left_biased_interpolate_yᵃᶜᵃ, 
        _left_biased_interpolate_yᵃᶠᵃ, 
        _right_biased_interpolate_yᵃᶜᵃ,
        _right_biased_interpolate_yᵃᶠᵃ

advection_schemes = [Centered, UpwindBiased, WENO]

@inline advective_order(buffer, ::Type{Centered}) = buffer * 2
@inline advective_order(buffer, AdvectionType)    = buffer * 2 - 1

function run_tracer_interpolation_test(c, ibg, scheme)

    for i in 6:19, j in 6:19
        if typeof(scheme) <: Centered
            @test CUDA.@allowscalar  _symmetric_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, c, 1) ≈ 1.0
        else    
            @test CUDA.@allowscalar  _left_biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, c, 1) ≈ 1.0
            @test CUDA.@allowscalar _right_biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, c, 1) ≈ 1.0
            @test CUDA.@allowscalar  _left_biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, scheme, c, 1) ≈ 1.0
            @test CUDA.@allowscalar _right_biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, scheme, c, 1) ≈ 1.0
        end
    end
end

function run_tracer_conservation_test(grid, scheme)

    model = HydrostaticFreeSurfaceModel(grid = grid, tracers = :c, 
                                        free_surface = ExplicitFreeSurface(),
                                        tracer_advection = scheme, 
                                        buoyancy = nothing,
                                        coriolis = nothing)

    c = model.tracers.c
    set!(model, c = 1.0)
    fill_halo_regions!(c)

    η = model.free_surface.η

    indices = model.grid isa ImmersedBoundaryGrid ? (5:7, 3:6, 1) : (2:5, 3:6, 1)

    interior(η, indices...) .= - 0.05
    fill_halo_regions!(η)

    wave_speed = sqrt(model.free_surface.gravitational_acceleration)
    dt = 0.1 / wave_speed
    for i in 1:10
        time_step!(model, dt)
    end

    @test maximum(c) ≈ 1.0 
    @test minimum(c) ≈ 1.0 
    @test mean(c)    ≈ 1.0

    return nothing
end

function run_momentum_interpolation_test(u, v, ibg, scheme)

    for i in 7:19, j in 7:19
        if typeof(scheme) <: Centered
            @test CUDA.@allowscalar  _symmetric_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, u, 1) ≈ 1.0
            @test CUDA.@allowscalar  _symmetric_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, v, 1) ≈ 1.0
            @test CUDA.@allowscalar  _symmetric_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, u, 1) ≈ 1.0
            @test CUDA.@allowscalar  _symmetric_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, v, 1) ≈ 1.0
        else    
            @test CUDA.@allowscalar  _left_biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, u, 1) ≈ 1.0
            @test CUDA.@allowscalar _right_biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, u, 1) ≈ 1.0
            @test CUDA.@allowscalar  _left_biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, u, 1) ≈ 1.0
            @test CUDA.@allowscalar _right_biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, u, 1) ≈ 1.0

            @test CUDA.@allowscalar  _left_biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, v, 1) ≈ 1.0
            @test CUDA.@allowscalar _right_biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, v, 1) ≈ 1.0
            @test CUDA.@allowscalar  _left_biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, v, 1) ≈ 1.0
            @test CUDA.@allowscalar _right_biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, v, 1) ≈ 1.0
        end
    end

    return nothing
end

for arch in archs
    @testset "Immersed tracer reconstruction" begin
        @info "Running immersed tracer reconstruction tests..."

        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y, z) -> (x < 5 || y < 5)))
    
        c = CenterField(ibg)
        set!(c, 1.0)
        wait(mask_immersed_field!(c))
        fill_halo_regions!(c)
    
        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))
            
            @info "  Testing immersed tracer reconstruction [$(typeof(arch)), $(summary(scheme))]"
            run_tracer_interpolation_test(c, ibg, scheme)
        end
    end

    @testset "Immersed tracer conservation" begin
        @info "Running immersed tracer conservation tests..."

        grid = RectilinearGrid(arch, size=(10, 8, 1), extent=(10, 8, 1), halo = (6, 6, 6), topology=(Bounded, Periodic, Bounded))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y, z) -> (x < 2)))
    
        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))
        
            for g in [grid, ibg]
                @info "  Testing immersed tracer conservation [$(typeof(arch)), $(summary(scheme)), $(typeof(g).name.wrapper)]"
                run_tracer_conservation_test(g, scheme)
            end
        end
    end

    @testset "Immersed momentum reconstruction" begin
        @info "Running immersed momentum recontruction tests..."

        grid = RectilinearGrid(arch, size=(20, 20), extent=(20, 20), topology=(Bounded, Bounded, Flat))
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y, z) -> (x < 5 || y < 5)))

        u = XFaceField(ibg)
        v = YFaceField(ibg)
        set!(u, 1.0)
        set!(v, 1.0)

        wait(mask_immersed_field!(u))
        wait(mask_immersed_field!(v))

        fill_halo_regions!(u)
        fill_halo_regions!(v)

        for adv in advection_schemes, buffer in [1, 2, 3, 4, 5]
            scheme = adv(order = advective_order(buffer, adv))
            
            @info "  Testing immersed momentum reconstruction [$(typeof(arch)), $(summary(adv))]"
            run_momentum_interpolation_test(u, v, scheme, ibg)
        end
    end
end

