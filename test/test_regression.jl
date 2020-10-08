using Oceananigans.Fields: interiorparent

function summarize_regression_test(fields, correct_fields)
    for (field_name, φ, φ_c) in zip(keys(fields), fields, correct_fields)
        Δ = φ .- φ_c

        Δ_min      = minimum(Δ)
        Δ_max      = maximum(Δ)
        Δ_mean     = mean(Δ)
        Δ_abs_mean = mean(abs, Δ)
        Δ_std      = std(Δ)

        matching    = sum(φ .≈ φ_c)
        grid_points = length(φ_c)

        @info @sprintf("Δ%s: min=%+.6e, max=%+.6e, mean=%+.6e, absmean=%+.6e, std=%+.6e (%d/%d matching grid points)",
                       field_name, Δ_min, Δ_max, Δ_mean, Δ_abs_mean, Δ_std, matching, grid_points)
    end
end

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

            field_names = [:ρ, :ρu, :ρv, :ρw]
            fields = [interior(model.total_density), interior(model.momenta.ρu),
                      interior(model.momenta.ρv), interior(model.momenta.ρw)[:, :, 1:end-1]]
            correct_fields = [file["ρ"], file["ρu"], file["ρv"], file["ρw"]]

            if tvar isa Energy
                push!(field_names, :ρe)
                push!(fields, interior(model.tracers.ρe))
                push!(correct_fields, file["ρe"])
            elseif tvar isa Entropy
                push!(field_names, :ρs)
                push!(fields, interior(model.tracers.ρs))
                push!(correct_fields, file["ρs"])
            end

            fields = NamedTuple{Tuple(field_names)}(Tuple(fields))
            correct_fields = NamedTuple{Tuple(field_names)}(Tuple(correct_fields))
            summarize_regression_test(fields, correct_fields)

            @test all(isapprox.(interior(model.momenta.ρu), file["ρu"], atol=1e-12))

            @test all(interior(model.total_density) .≈ file["ρ"])
            @test all(interior(model.momenta.ρv)    .≈ file["ρv"])
            @test all(interior(model.momenta.ρw)[:, :, 1:end-1]    .≈ file["ρw"])
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

            field_names = [:ρ, :ρu, :ρv, :ρw, :ρ₁, :ρ₂, :ρ₃]
            fields = [interior(model.total_density), interior(model.momenta.ρu),
                      interior(model.momenta.ρv), interior(model.momenta.ρw)[:, :, 1:end-1],
                      interior(model.tracers.ρ₁), interior(model.tracers.ρ₂),
                      interior(model.tracers.ρ₃)]
            correct_fields = [file["ρ"], file["ρu"], file["ρv"], file["ρw"],
                              file["ρ₁"], file["ρ₂"], file["ρ₃"]]

            if tvar isa Energy
                push!(field_names, :ρe)
                push!(fields, interior(model.tracers.ρe))
                push!(correct_fields, file["ρe"])
            elseif tvar isa Entropy
                push!(field_names, :ρs)
                push!(fields, interior(model.tracers.ρs))
                push!(correct_fields, file["ρs"])
            end

            fields = NamedTuple{Tuple(field_names)}(Tuple(fields))
            correct_fields = NamedTuple{Tuple(field_names)}(Tuple(correct_fields))
            summarize_regression_test(fields, correct_fields)

            @test all(isapprox.(interior(model.momenta.ρu), file["ρu"], atol=1e-12))

            @test all(interior(model.total_density) .≈ file["ρ"])
            @test all(interior(model.momenta.ρv)    .≈ file["ρv"])
            @test all(interior(model.momenta.ρw)[:, :, 1:end-1]    .≈ file["ρw"])
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
