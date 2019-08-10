@testset "Forcings" begin
    println("Testing forcings...")

    add_one(args...) = 1.0

    function test_forcing(fld)
        kwarg = Dict(Symbol(:F, fld)=>add_one)
        forcing = Forcing(; kwarg...)
        f = getfield(forcing, fld)
        f() == 1.0
    end

    for fld in (:u, :v, :w, :T, :S)
        @test test_forcing(fld)
    end
end
