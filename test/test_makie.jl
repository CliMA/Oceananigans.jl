include("dependencies_for_runtests.jl")

using CairoMakie

function test_moviemaker(arch)
    # Create a simple model for testing
    grid = RectilinearGrid(arch, size=(4, 4), extent=(1, 1), topology=(Periodic, Periodic, Flat))
    model = NonhydrostaticModel(; grid)

    # Set up a simple simulation
    simulation = Simulation(model, Δt=1, stop_time=5)

    fig1 = Figure()
    ax1 = Axis(fig1[1, 1])

    update_plot(sim, fig) = heatmap!(fig[1, 1], model.velocities.u)
    mm = MovieMaker(fig1, update_plot, filename="movie1.mp4")
    add_callback!(simulation, mm, IterationInterval(1))

    @test mm isa MovieMaker
    @test mm.figure === fig1
    @test mm.io isa VideoStream

    fig2 = Figure()
    ax2 = Axis(fig2[1, 1])

    add_movie_maker!(simulation, IterationInterval(1), fig2, update_plot; filename="movie2.mp4")

    run!(simulation)

    # Check if files were created
    @test isfile("movie1.mp4")
    rm("movie1.mp4")

    @test isfile("movie2.mp4")
    rm("movie2.mp4")

    return nothing
end

for arch in archs
    @testset "Makie extension tests [$(typeof(arch))]..." begin
        @info "Testing Makie extension [$(typeof(arch))]..."
        test_moviemaker(arch)
    end
end
