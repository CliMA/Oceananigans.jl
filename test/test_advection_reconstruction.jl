include("dependencies_for_runtests.jl")

using Random

using Oceananigans.Fields: instantiate
using Oceananigans.Advection: LeftBias, RightBias

using Oceananigans.Advection: compute_face_reduced_order_x,
                              compute_face_reduced_order_y,
                              compute_face_reduced_order_z,
                              compute_center_reduced_order_x,
                              compute_center_reduced_order_y,
                              compute_center_reduced_order_z

using Oceananigans.Advection: symmetric_interpolate_xᶠᵃᵃ,
                              symmetric_interpolate_yᵃᶠᵃ,
                              symmetric_interpolate_zᵃᵃᶠ,
                              symmetric_interpolate_xᶜᵃᵃ,
                              symmetric_interpolate_yᵃᶜᵃ,
                              symmetric_interpolate_zᵃᵃᶜ

using Oceananigans.Advection: biased_interpolate_xᶠᵃᵃ,
                              biased_interpolate_yᵃᶠᵃ,
                              biased_interpolate_zᵃᵃᶠ,
                              biased_interpolate_xᶜᵃᵃ,
                              biased_interpolate_yᵃᶜᵃ,
                              biased_interpolate_zᵃᵃᶜ

@inline grid_args(::Tuple{Bounded, Flat, Flat}) = (; x = (0, 1))
@inline grid_args(::Tuple{Flat, Bounded, Flat}) = (; y = (0, 1))
@inline grid_args(::Tuple{Flat, Flat, Bounded}) = (; z = (0, 1))

red_order_field(grid::AbstractGrid{<:Any, <:Bounded, <:Flat, <:Flat}, adv, ::Face) = 
    compute!(Field(KernelFunctionOperation{Face, Nothing, Nothing}(compute_face_reduced_order_x, grid, adv)))

red_order_field(grid::AbstractGrid{<:Any, <:Flat, <:Bounded, <:Flat}, adv, ::Face) = 
    compute!(Field(KernelFunctionOperation{Nothing, Face, Nothing}(compute_face_reduced_order_y, grid, adv)))

red_order_field(grid::AbstractGrid{<:Any, <:Flat, <:Flat, <:Bounded}, adv, ::Face) = 
    compute!(Field(KernelFunctionOperation{Nothing, Nothing, Face}(compute_face_reduced_order_z, grid, adv)))

red_order_field(grid::AbstractGrid{<:Any, <:Bounded, <:Flat, <:Flat}, adv, ::Center) = 
    compute!(Field(KernelFunctionOperation{Center, Nothing, Nothing}(compute_center_reduced_order_x, grid, adv)))

red_order_field(grid::AbstractGrid{<:Any, <:Flat, <:Bounded, <:Flat}, adv, ::Center) = 
    compute!(Field(KernelFunctionOperation{Nothing, Center, Nothing}(compute_center_reduced_order_y, grid, adv)))

red_order_field(grid::AbstractGrid{<:Any, <:Flat, <:Flat, <:Bounded}, adv, ::Center) = 
    compute!(Field(KernelFunctionOperation{Nothing, Nothing, Center}(compute_center_reduced_order_z, grid, adv)))

@testset "Reduced order reconstruction" begin

    for topology in ((Bounded, Flat, Flat), 
                     (Flat, Bounded, Flat),
                     (Flat, Flat, Bounded))

        extent = grid_args(instantiate.(topology))
        grid = RectilinearGrid(; size = 10, extent..., topology, halo=5)
        adv  = Centered(order=10) 

        red_ord_face   = red_order_field(grid, adv, Face())
        red_ord_center = red_order_field(grid, adv, Center())

        @test all(interior(red_ord_face)[:]   .== [1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1])
        @test all(interior(red_ord_center)[:] .== [1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
    
        ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary(false)) # A fake immersed boundary

        red_ord_face   = red_order_field(ibg, adv, Face())
        red_ord_center = red_order_field(ibg, adv, Center())

        @test all(interior(red_ord_face)[:]   .== [1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1])
        @test all(interior(red_ord_center)[:] .== [1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
    end

    grid = RectilinearGrid(size=(20, 20, 20), extent=(1, 1, 1), halo=(6, 6, 6), topology=(Periodic, Periodic, Periodic))
    c = CenterField(grid)    
    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)

    Random.seed!(1234)
    set!(c, (x, y, z) -> rand())
    set!(u, (x, y, z) -> rand())
    set!(v, (x, y, z) -> rand())
    set!(w, (x, y, z) -> rand())

    fill_halo_regions!((c, u, v, w))

    Nx, Ny, Nz = size(grid)

    # Test symmetric reconstruction (Centered schemes)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        scheme   = Centered(order=10)
        rscheme1 = Centered(order=8)
        rscheme2 = Centered(order=6)
        rscheme3 = Centered(order=4)
        rscheme4 = Centered(order=2)

        for s in (scheme, rscheme1, rscheme2, rscheme3, rscheme4)
            @test symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 1, c) == symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme4, 1, c)
            @test symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 1, c) == symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme4, 1, c)
            @test symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 1, c) == symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme4, 1, c)
        end
        
        for s in (scheme, rscheme1, rscheme2, rscheme3)
            @test symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 2, c) == symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme3, 2, c)
            @test symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 2, c) == symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme3, 2, c)
            @test symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 2, c) == symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme3, 2, c)
        end

        for s in (scheme, rscheme1, rscheme2)
            @test symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 3, c) == symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme2, 3, c)
            @test symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 3, c) == symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme2, 3, c)
            @test symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 3, c) == symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme2, 3, c)
        end

        for s in (scheme, rscheme1)
            @test symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 4, c) == symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme1, 4, c)
            @test symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 4, c) == symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme1, 4, c)
            @test symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 4, c) == symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme1, 4, c)
        end

        scheme   = UpwindBiased(order=9)
        rscheme1 = UpwindBiased(order=7)
        rscheme2 = UpwindBiased(order=5)
        rscheme3 = UpwindBiased(order=3)
        rscheme4 = UpwindBiased(order=1)
        
        for bias in (LeftBias(), RightBias())
            for s in (scheme, rscheme1, rscheme2, rscheme3, rscheme4)
                @test biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 1, bias, c) == biased_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme4, 1, bias, c)
                @test biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 1, bias, c) == biased_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme4, 1, bias, c)
                @test biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 1, bias, c) == biased_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme4, 1, bias, c)
            end
            
            for s in (scheme, rscheme1, rscheme2, rscheme3)
                @test biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 2, bias, c) == biased_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme3, 2, bias, c)
                @test biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 2, bias, c) == biased_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme3, 2, bias, c)
                @test biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 2, bias, c) == biased_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme3, 2, bias, c)
            end

            for s in (scheme, rscheme1, rscheme2)
                @test biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 3, bias, c) == biased_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme2, 3, bias, c)
                @test biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 3, bias, c) == biased_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme2, 3, bias, c)
                @test biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 3, bias, c) == biased_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme2, 3, bias, c)
            end

            for s in (scheme, rscheme1)
                @test biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 4, bias, c) == biased_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme1, 4, bias, c)
                @test biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 4, bias, c) == biased_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme1, 4, bias, c)
                @test biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 4, bias, c) == biased_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme1, 4, bias, c)
            end
        end

        scheme   = WENO(order=9)
        rscheme1 = WENO(order=7)
        rscheme2 = WENO(order=5)
        rscheme3 = WENO(order=3)
        
        for bias in (LeftBias(), RightBias())
            for s in (scheme, rscheme1, rscheme2, rscheme3)
                @test biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 2, bias, c) ≈ biased_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme3, 2, bias, c)
                @test biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 2, bias, c) ≈ biased_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme3, 2, bias, c)
                @test biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 2, bias, c) ≈ biased_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme3, 2, bias, c)
            end

            for s in (scheme, rscheme1, rscheme2)
                @test biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 3, bias, c) ≈ biased_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme2, 3, bias, c)
                @test biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 3, bias, c) ≈ biased_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme2, 3, bias, c)
                @test biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 3, bias, c) ≈ biased_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme2, 3, bias, c)
            end

            for s in (scheme, rscheme1)
                @test biased_interpolate_xᶠᵃᵃ(i, j, k, grid, s, 4, bias, c) ≈ biased_interpolate_xᶠᵃᵃ(i, j, k, grid, rscheme1, 4, bias, c)
                @test biased_interpolate_yᵃᶠᵃ(i, j, k, grid, s, 4, bias, c) ≈ biased_interpolate_yᵃᶠᵃ(i, j, k, grid, rscheme1, 4, bias, c)
                @test biased_interpolate_zᵃᵃᶠ(i, j, k, grid, s, 4, bias, c) ≈ biased_interpolate_zᵃᵃᶠ(i, j, k, grid, rscheme1, 4, bias, c)
            end
        end
    end

end

