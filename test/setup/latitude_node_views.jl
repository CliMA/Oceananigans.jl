using Test
using MPI
using Oceananigans
using Oceananigans.Grids: φnodes, φnode, λnodes, λnode
using Oceananigans.DistributedComputations: Distributed, Partition

function test_latitude_node_views(communicator)
    rank = MPI.Comm_rank(communicator)
    ranks = MPI.Comm_size(communicator)
    latitude_architecture = Distributed(CPU(); partition=Partition(y=ranks), communicator)
    longitude_architecture = Distributed(CPU(); partition=Partition(x=ranks), communicator)

    @testset "One-dimensional coordinate views on rank $rank" begin
        for FT in (Float32, Float64)
            N = 4ranks
            stretched_latitudes = [FT(-80) + FT(160) * (FT(j) / FT(N))^2 for j in 0:N]
            latitude_cases = ((:regular, (-80, 80)), (:stretched, stretched_latitudes))
            for (latitude_case, latitude) in latitude_cases, halo in (nothing, (1, 2, 1), (2, 1, 1))
                @testset "$latitude_case latitude [$FT, halo=$halo]" begin
                    latitude_grid = LatitudeLongitudeGrid(latitude_architecture, FT;
                                                         size=(4, 4ranks, 1), halo,
                                                         longitude=(0, 360), latitude, z=(-1, 0))
                    serial_grid = LatitudeLongitudeGrid(CPU(), FT;
                                                       size=(4, 4ranks, 1), halo,
                                                       longitude=(0, 360), latitude, z=(-1, 0))
                    longitude_grid = LatitudeLongitudeGrid(longitude_architecture, FT;
                                                          size=(4ranks, 4ranks, 1), halo,
                                                          longitude=(0, 360), latitude, z=(-1, 0))
                    serial_longitude_grid = LatitudeLongitudeGrid(CPU(), FT;
                                                                 size=(4ranks, 4ranks, 1), halo,
                                                                 longitude=(0, 360), latitude, z=(-1, 0))

                    for location in (Center(), Face())
                        latitude_count = size(latitude_grid, 2) + (location isa Face && rank == ranks - 1)
                        latitude_indices = 1:latitude_count
                        global_latitude_indices = rank * size(latitude_grid, 2) .+ latitude_indices
                        latitude_values = φnodes(latitude_grid, location)
                        scalar_latitudes = [φnode(j, latitude_grid, location) for j in latitude_indices]
                        serial_latitudes = φnodes(serial_grid, location)
                        expected_latitudes = serial_latitudes[global_latitude_indices]
                        latitude_storage = location isa Center ? latitude_grid.φᵃᶜᵃ : latitude_grid.φᵃᶠᵃ

                        @test length(latitude_values) == latitude_count
                        @test collect(latitude_values) == scalar_latitudes
                        @test collect(latitude_values) == collect(expected_latitudes)
                        @test eltype(latitude_values) == FT
                        if latitude_case == :regular
                            @test latitude_values isa AbstractRange
                        else
                            @test latitude_values isa SubArray
                            @test Base.mightalias(latitude_values, parent(latitude_storage))
                        end
                        @test collect(φnodes(latitude_grid, location; with_halos=true)) == collect(latitude_storage)
                        @test φnodes(latitude_grid, nothing) === nothing

                        selected_indices = 2:min(3, latitude_count)
                        selected_values = φnodes(latitude_grid, location; indices=selected_indices)
                        @test collect(selected_values) == scalar_latitudes[selected_indices]
                        @test collect(selected_values) == collect(expected_latitudes[selected_indices])
                        if latitude_case == :regular
                            @test selected_values isa AbstractRange
                        else
                            @test Base.mightalias(selected_values, parent(latitude_storage))
                        end
                        @test collect(serial_latitudes) == [φnode(j, serial_grid, location) for j in eachindex(serial_latitudes)]

                        longitude_count = size(longitude_grid, 1)
                        longitude_indices = 1:longitude_count
                        global_longitude_indices = rank * longitude_count .+ longitude_indices
                        longitude_values = λnodes(longitude_grid, location)
                        @test length(longitude_values) == longitude_count
                        @test collect(longitude_values) == [λnode(i, longitude_grid, location) for i in longitude_indices]
                        @test collect(longitude_values) == collect(λnodes(serial_longitude_grid, location)[global_longitude_indices])

                        longitude_latitude_count = size(longitude_grid, 2) + (location isa Face)
                        longitude_latitude_values = φnodes(longitude_grid, location)
                        longitude_scalar_latitudes = [φnode(j, longitude_grid, location) for j in 1:longitude_latitude_count]
                        serial_longitude_latitudes = φnodes(serial_longitude_grid, location)
                        @test collect(longitude_latitude_values) == longitude_scalar_latitudes
                        @test collect(longitude_latitude_values) == collect(serial_longitude_latitudes)
                    end
                end
            end
        end
    end
    return nothing
end
