
using Oceananigans
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation, ConservativeFormulation, shallow_water_velocities
using LinearAlgebra: norm
using NCDatasets, Printf, CairoMakie
using Polynomials: fit
using Random

Lx, Ly, Lz = 2π, 20, 10
Nx, Ny = 128, 128

grid = RectilinearGrid(GPU(), size = (Nx, Ny),
                       x = (0, Lx), y = (-Ly/2, Ly/2),
                       topology = (Periodic, Bounded, Flat))

gravitational_acceleration = 1
coriolis = FPlane(f=1)

solution = Dict()

@inline weno_advection(::Val{:conservative}, order)     = WENO(; order)
@inline weno_advection(::Val{:vorticitystencil}, order) = WENO(; order, vector_invariant = VorticityStencil())
@inline weno_advection(::Val{:velocitystencil}, order)  = WENO(; order, vector_invariant = VelocityStencil())

@inline Formulation(::Val{:conservative})     = ConservativeFormulation()
@inline Formulation(::Val{:vorticitystencil}) = VectorInvariantFormulation()
@inline Formulation(::Val{:velocitystencil})  = VectorInvariantFormulation()

for form in [:conservative, :vorticitystencil, :velocitystencil]
    for order in [3, 5, 7, 9, 11]

        @info "starting with $form and order $order"

        adv = weno_advection(Val(form), order)
        model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                                    mass_advection = WENO(; order),
                                    momentum_advection = adv,
                                    formulation = Formulation(Val(form)))

        U = 1 # Maximum jet velocity
        f = coriolis.f
        g = gravitational_acceleration
        Δη = f * U / g  # Maximum free-surface deformation as dictated by geostrophy

        h̄(x, y, z) = Lz - Δη * tanh(y)
        ū(x, y, z) = U * sech(y)^2
        ω̄(x, y, z) = 2 * U * sech(y)^2 * tanh(y)

        small_amplitude = 1e-4

        Random.seed!(123)
        uⁱ(x, y, z) = ū(x, y, z) + small_amplitude * exp(-y^2) * rand()
        uhⁱ(x, y, z) = uⁱ(x, y, z) * h̄(x, y, z)

        ū̄h(x, y, z) = ū(x, y, z) * h̄(x, y, z)

        if model.formulation isa ConservativeFormulation
            set!(model, uh = ū̄h, h = h̄)
        else
            set!(model, u = ū, h = h̄)
        end

        h    = model.solution.h
        u, v = shallow_water_velocities(model) 

        ## Build and compute mean vorticity discretely
        ω = Field(∂x(v) - ∂y(u))
        compute!(ω)

        ## Copy mean vorticity to a new field
        ωⁱ = Field{Face, Face, Nothing}(model.grid)
        ωⁱ .= ω

        ## Use this new field to compute the perturbation vorticity
        ω′ = Field(ω - ωⁱ)

        # and finally set the "true" initial condition with noise,

        if model.formulation isa ConservativeFormulation
            set!(model, uh = uhⁱ)
        else
            set!(model, u = uⁱ)
        end

        simulation = Simulation(model, Δt = 1e-2, stop_time = 150)

        perturbation_norm(args...) = norm(v)

        fields_filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_fields.nc")
        simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; ω, ω′),
                                                                filename = fields_filename,
                                                                schedule = TimeInterval(1),
                                                                overwrite_existing = true)

        # Build the `output_writer` for the growth rate, which is a scalar field.
        # Output every time step.

        growth_filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_perturbation_norm.nc")
        simulation.output_writers[:growth] = NetCDFOutputWriter(model, (; perturbation_norm),
                                                                filename = growth_filename,
                                                                schedule = IterationInterval(1),
                                                                dimensions = (; perturbation_norm = ()),
                                                                overwrite_existing = true)

        # And finally run the simulation.
        run!(simulation)
        
        # Read in the `output_writer` for the scalar field (the norm of ``v``-velocity).

        ds2 = NCDataset(simulation.output_writers[:growth].filepath, "r")

            t = ds2["time"][:]
        norm_v = ds2["perturbation_norm"][:]

        close(ds2)

        I = 6000:7000

        degree = 1
        linear_fit_polynomial = fit(t[I], log.(norm_v[I]), degree, var = :t)

        # We can get the coefficient of the ``n``-th power from the fitted polynomial by using `n` 
        # as an index, e.g.,

        constant, slope = linear_fit_polynomial[0], linear_fit_polynomial[1]

        # We then use the computed linear fit coefficients to construct the best fit and plot it 
        # together with the time-series for the perturbation norm for comparison. 

        best_fit = @. exp(constant + slope * t)

        fig = Figure()
        ax  = Axis(fig[1, 1], yscale = log10,
        limits = (nothing, (1e-3, 30)),
        xlabel = "time",
        ylabel = "norm(v)",
        title = "growth of perturbation norm")

        lines!(ax, t, norm_v;
            linewidth = 4,
            label = "norm(v)")

        lines!(ax, t[I], 2 * best_fit[I]; # factor 2 offsets fit from curve for better visualization
            linewidth = 4,
            label = "best fit")

        axislegend(position = :rb)

        save("fig_$(form)_$(order).png", fig)

        # The slope of the best-fit curve on a logarithmic scale approximates the rate at which instability
        # grows in the simulation. Let's see how this compares with the theoretical growth rate.

        println("Numerical growth rate is approximated to be ", slope, ",\n",
                "which is very close to the theoretical value of 0.139.")
                
        solution[(form, order)] = slope
    end
end

using PrettyTables

cases = [(:Conservative for i in 1:5)..., (:VI_Vorticity for i in 1:5)..., (:VI_Velocity for i in 1:5)... ]
order = [3, 5, 7, 9, 11, 3, 5, 7, 9, 11, 3, 5, 7, 9, 11]
outcome = [0.13900223349731414, 0.13873604501136147, (0.13873604585518295 for j in 1:3)..., (1.0 for i in 1:10)...]
diff = abs.(0.139 .- outcome) ./ 0.139

data = hcat(cases, order, outcome, diff)

header = (["Formulation", "Order", "Growth rate", "Error"])
pretty_table(data; header, tf = tf_unicode_rounded, header_crayon = crayon"yellow bold") #formatters = ft_printf("%.2e", 3:4))
