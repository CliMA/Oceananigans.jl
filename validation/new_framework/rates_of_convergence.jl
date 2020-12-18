using Plots
using LaTeXStrings
using Printf
using Polynomials

using Oceananigans
using Oceananigans.Advection

### Model parameters and function

           U  = 1
           L  = 2.5
           W  = 0.1
           Ns = 2 .^ (4:7)

           Δt = 0.01 * minimum(L/Ns) / U

c(x, t, U, W) = exp( -(x - U * t)^2 / W );

### Difference Operator

@inline δx(i, c) = @inbounds c[i+1] - c[i]

### Time Stepping Schemes

struct ForwardEuler end
struct AdamsBashforth2 end

Time_Stepper(i, c, F, F₋₁, Δx, Δt, ::ForwardEuler)    = c[i] - Δt/Δx*(     δx(i, F)              )
Time_Stepper(i, c, F, F₋₁, Δx, Δt, ::AdamsBashforth2) = c[i] - Δt/Δx*( 3 * δx(i, F) - δx(i, F₋₁) )/2

time_steppers = (
    ForwardEuler,
    AdamsBashforth2
)

### Advection Schemes

struct UpwindBiasedFirstOrder    end
struct CenteredSixthOrder  end

advective_flux(i, j, k, grid, ::UpwindBiasedFirstOrder,    U, c) =                 c[i-1]
advective_flux(i, j, k, grid, ::CenteredSecondOrder,       U, c) = (   c[i]   +    c[i-1]) / 2
advective_flux(i, j, k, grid, ::UpwindBiasedThirdOrder,    U, c) = ( 2*c[i]   +  5*c[i-1]    -  c[i-2]) / 6   
advective_flux(i, j, k, grid, ::CenteredFourthOrder,       U, c) = ( 7(c[i]   +    c[i-1] )  - (c[i+1] +    c[i-2]) ) / 12
advective_flux(i, j, k, grid, ::UpwindBiasedFifthOrder,    U, c) = (-3*c[i+1] + 27*c[i]    + 47*c[i-1] - 13*c[i-2] + 2*c[i-3] ) / 60
advective_flux(i, j, k, grid, ::CenteredSixthOrder,        U, c) = (37(c[i] +      c[i-1] ) - 8(c[i+1]    + c[i-2]) + (c[i+2] + c[i-3]) ) / 60

schemes = (
    UpwindBiasedFirstOrder, 
    CenteredSecondOrder, 
    UpwindBiasedThirdOrder, 
    CenteredFourthOrder, 
    UpwindBiasedFifthOrder, 
    CenteredSixthOrder
);

### Dictionaries and Functions for output

rate_of_convergence(::UpwindBiasedFirstOrder) = 1
rate_of_convergence(::CenteredSecondOrder)    = 2
rate_of_convergence(::UpwindBiasedThirdOrder) = 3
rate_of_convergence(::CenteredFourthOrder)    = 4
rate_of_convergence(::UpwindBiasedFifthOrder) = 5
rate_of_convergence(::CenteredSixthOrder)     = 6

labels(::UpwindBiasedFirstOrder) = "Upwind1ˢᵗ"
labels(::CenteredSecondOrder)    = "Center2ⁿᵈ"
labels(::UpwindBiasedThirdOrder) = "Upwind3ʳᵈ"
labels(::CenteredFourthOrder)    = "Center4ᵗʰ"
labels(::UpwindBiasedFifthOrder) = "Upwind5ᵗʰ"
labels(::CenteredSixthOrder)     = "Center6ᵗʰ"

shapes(::UpwindBiasedFirstOrder) = :circle
shapes(::CenteredSecondOrder)    = :diamond
shapes(::UpwindBiasedThirdOrder) = :dtriangle
shapes(::CenteredFourthOrder)    = :rect
shapes(::UpwindBiasedFifthOrder) = :star5
shapes(::CenteredSixthOrder)     = :star6

colors(::UpwindBiasedFirstOrder) = :blue
colors(::CenteredSecondOrder)    = :green
colors(::UpwindBiasedThirdOrder) = :red
colors(::CenteredFourthOrder)    = :cyan
colors(::UpwindBiasedFifthOrder) = :magenta
colors(::CenteredSixthOrder)     = :purple

error  = Dict()
ROC    = Dict()

time_stepper = AdamsBashforth2

for N in Ns, scheme in schemes

    local grid = RegularCartesianGrid(size=(N, 1, 1), x=(-1, -1+L), y=(0, 1), z=(0, 1))
    xC = reshape(grid.xC, length(grid.xC), 1, 1)

    local c₋₁ = c.(xC, -Δt, U, W);
    local c₀  = c.(xC,   0, U, W);
    local c₁  = c.(xC,  Δt, U, W);

    local F₀ = zeros(N+2,1,1)
    local F₋₁= zeros(N+2,1,1)

    local Ftmp = zeros(N+2,1,1)
    
    for i in 4:N-2, j in 1:grid.Ny, k in 1:grid.Nz
        #F₀[i, j, k] = advective_tracer_flux_x(i, j, k, grid, scheme(), U, c₀)
        #F₋₁[i, j, k] = advective_tracer_flux_x(i, j, k, grid, scheme(), U, c₋₁)
        F₀[i, j, k] = advective_flux(i, j, k, grid, scheme(), U, c₀)
        F₋₁[i, j, k] = advective_flux(i, j, k, grid, scheme(), U, c₋₁)
    end

    local cₛᵢₘ = zeros(N)
    local cₑᵣᵣ = zeros(N)

    for i in 4:N-1
        
        cₛᵢₘ[i] = Time_Stepper(i, c₀, F₀, F₋₁, grid.Δx, Δt, time_stepper())
        cₑᵣᵣ[i] = cₛᵢₘ[i] - c₁[i]
        
    end
    
    error[(N, scheme)] = maximum(abs.(cₑᵣᵣ))

end

### Compute rates of convergence

for scheme in schemes
    
    name = labels(scheme())
    roc = rate_of_convergence(scheme())
    j = 3
    local p = fit(log10.(Ns[2:end]), log10.([error[(N, scheme)] for N in Ns][2:end]), 1)
    ROC[scheme] = p[1]
    println("Method = ", name, ", Rate of Convergence = ", @sprintf("%.2f", -ROC[scheme]), ", Expected = ", roc)
    
end

### Makes Plot

plt = plot()

for scheme in schemes
    
   plot!(
        plt,
        log2.(Ns),
        [error[(N, scheme)] for N in Ns],
        seriestype = :scatter,
             shape = shapes(scheme()),
        markersize = 6,
       markercolor = colors(scheme()),
            xscale = :log10,
            yscale = :log10,
            xlabel = "log₂N",
            xticks = (log2.(Ns), string.(Int.(log2.(Ns)))),
            ylabel = "L∞-norm: |cₛᵢₘ - c₁|",
             label =  string(labels(scheme()))*" slope = "*@sprintf("%.2f", ROC[scheme]),
            legend = :outertopright,
             title = "Rates of Convergence"
        )

end

for scheme in schemes
        
    roc = rate_of_convergence(scheme())
    
    plot!(
        plt,
        log2.(Ns[end-3:end]),
        [error[(N, scheme)] for N in Ns][end-3] .* (Ns[end-3] ./ Ns[end-3:end]) .^ roc,
        linestyle = :solid,
               lw = 3,
        linecolor = colors(scheme()),
            label = "Expected slope = "*@sprintf("%d",-roc)
    )
    
end

display(plt)
savefig(plt, "convergence_rates")


