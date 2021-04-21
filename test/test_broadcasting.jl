@testset "Field broadcasting" begin
    @info "  Testing broadcasting with fields..."

    for arch in archs
        grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
        a, b, c = [CenterField(arch, grid) for i = 1:3]

        a .= 1
        @test all(a .== 1) 

        b .= 2
        c .= a .+ b .+ 1
        @test all(c .== 4) 

        r, p, q = [ReducedField(Center, Center, Nothing, arch, grid, dims=3) for i = 1:3]

        r .= 2 
        @test all(r .== 2) 

        p .= 3 
        q .= r .* p .+ 1
        @test all(q .== 7) 
    end
end

