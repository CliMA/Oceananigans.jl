using Oceananigans.Fields: interiorparent

@testset "Regression" begin
    @info "Testing regression..."

    include("../verification/dry_rising_thermal_bubble/dry_rising_thermal_bubble.jl")

    for tvar in (Energy(), Entropy())
        @testset "Dry rising thermal bubble [$(typeof(tvar))]" begin
            @info "  Testing dry rising thermal bubble regression [$(typeof(tvar))]..."

            simulation = simulate_dry_rising_thermal_bubble(
                end_time=4.999, thermodynamic_variable=tvar, make_plots=false)

            model = simulation.model
            regression_filepath = "thermal_bubble_regression_$(typeof(tvar)).jld2"

            # UNCOMMENT TO GENERATE REGRESSION DATA!
            # jldopen(regression_filepath, "w") do file
            #     file["ρ"]  = interiorparent(model.total_density)
            #     file["ρu"] = interiorparent(model.momenta.ρu)
            #     file["ρv"] = interiorparent(model.momenta.ρv)
            #     file["ρw"] = interiorparent(model.momenta.ρw)
            #
            #     if tvar isa Energy
            #         file["ρe"] = interiorparent(model.tracers.ρe)
            #     elseif tvar isa Entropy
            #         file["ρs"] = interiorparent(model.tracers.ρs)
            #     end
            # end

            file = jldopen(regression_filepath, "r")
            @test all(interior(model.total_density) .≈ file["ρ"])
            @test all(interior(model.momenta.ρu)    .≈ file["ρu"])
            @test all(interior(model.momenta.ρv)    .≈ file["ρv"])
            @test all(interior(model.momenta.ρw)    .≈ file["ρw"])
            if tvar isa Energy
                @test all(interior(model.tracers.ρe) .≈ file["ρe"])
            elseif tvar isa Entropy
                @test all(interior(model.tracers.ρs) .≈ file["ρs"])
            end
        end
    end

    include("../verification/three_gas_dry_rising_thermal_bubble/three_gas_dry_rising_thermal_bubble.jl")

    for tvar in (Energy(), Entropy())
        @testset "Three gas thermal bubble [$(typeof(tvar))]" begin
            @info "  Testing three gas thermal bubble regression [$(typeof(tvar))]..."

            simulation = simulate_three_gas_dry_rising_thermal_bubble(
                end_time=4.999, thermodynamic_variable=tvar, make_plots=false)

            model = simulation.model
            regression_filepath = "three_gas_thermal_bubble_regression_$(typeof(tvar)).jld2"

            # UNCOMMENT TO GENERATE REGRESSION DATA!
            # jldopen(regression_filepath, "w") do file
            #     file["ρ"]  = interiorparent(model.total_density)
            #     file["ρu"] = interiorparent(model.momenta.ρu)
            #     file["ρv"] = interiorparent(model.momenta.ρv)
            #     file["ρw"] = interiorparent(model.momenta.ρw)
            #     file["ρ₁"] = interiorparent(model.tracers.ρ₁)
            #     file["ρ₂"] = interiorparent(model.tracers.ρ₂)
            #     file["ρ₃"] = interiorparent(model.tracers.ρ₃)
            #
            #     if tvar isa Energy
            #         file["ρe"] = interiorparent(model.tracers.ρe)
            #     elseif tvar isa Entropy
            #         file["ρs"] = interiorparent(model.tracers.ρs)
            #     end
            # end

            file = jldopen(regression_filepath, "r")
            @test all(interior(model.total_density) .≈ file["ρ"])
            @test all(interior(model.momenta.ρu)    .≈ file["ρu"])
            @test all(interior(model.momenta.ρv)    .≈ file["ρv"])
            @test all(interior(model.momenta.ρw)    .≈ file["ρw"])
            @test all(interior(model.tracers.ρ₁)    .≈ file["ρ₁"])
            @test all(interior(model.tracers.ρ₂)    .≈ file["ρ₂"])
            @test all(interior(model.tracers.ρ₃)    .≈ file["ρ₃"])

            if tvar isa Energy
                @test all(interior(model.tracers.ρe) .≈ file["ρe"])
            elseif tvar isa Entropy
                @test all(interior(model.tracers.ρs) .≈ file["ρs"])
            end
        end
    end
end
