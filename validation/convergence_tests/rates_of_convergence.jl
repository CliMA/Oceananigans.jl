using Plots
using LaTeXStrings
using Printf

   U  = 1
   L  = 2.5
   W  = 0.1
   Ns = 2 .^ (4:7)

   Δt = 0.01 * minimum(L/Ns) / U /1e3

error = Dict()

struct UpwindFirstOrder end
struct CenteredSecondOrder end
struct UpwindThirdOrder end
struct CenteredForthOrder end
struct UpwindFifthOrder end
struct CenteredSixthOrder end

Flux(c, ::UpwindFirstOrder)    =     c[3:end-3]
Flux(c, ::CenteredSecondOrder) = (   c[3:end-3] +    c[4:end-2]) / 2
Flux(c, ::UpwindThirdOrder)    = ( 2*c[4:end-2] +  5*c[3:end-3]  - c[2:end-4])/6   
Flux(c, ::CenteredForthOrder)  = ( 7(c[3:end-3] +    c[4:end-2] ) - (c[2:end-4] + c[5:end-1]) )/12
Flux(c, ::UpwindFifthOrder)    = (-3*c[5:end-1] + 27*c[4:end-2] + 47*c[3:end-3] - 13*c[2:end-4] + 2*c[1:end-5] )/60
Flux(c, ::CenteredSixthOrder)  = (37(c[3:end-3] +    c[4:end-2] ) - 8(c[2:end-4] + c[5:end-1]) + (c[1:end-5] + c[6:end]) )/60

schemes = (
    UpwindFirstOrder, 
    CenteredSecondOrder, 
    UpwindThirdOrder, 
    CenteredForthOrder, 
    UpwindFifthOrder, 
    CenteredSixthOrder
);

methods = (
    "Upwind1ˢᵗ",
    "Center2ⁿᵈ",
    "Upwind3ʳᵈ",
    "Center4ᵗʰ",
    "Upwind5ᵗʰ",
    "Center6ᵗʰ"
    );
orders = (1, 2, 3, 4, 5, 6)

roc    = zeros(length(methods))
labels = Dict()

for i in 1:length(schemes)
           scheme = schemes[i]
          roc[i]  = i
   labels[scheme] = methods[i] 
end

# Advection of a Gaussian.
c(x, t, U, W) = exp(-(x - U * t)^2/W);

for N in Ns, scheme in schemes
    
    Δx = L/N
    
    xf = range(-1, L - 1, step = Δx);
    xc = range(-1 .+ Δx/2, L-1-Δx/2, step = Δx)
    
    c₀ = c.(xc, 0, U, W);
    cexact = c.(xc, Δt, U, W);
    
    F = Flux(c₀, scheme())
        
    cnew = c₀[4:end-3] - Δt/Δx*(F[2:end] - F[1:end-1]);
    
    error[(N, scheme)] = maximum(abs.(cnew - cexact[4:end-3]))

end

plt = plot()

for scheme in schemes
    
   plot!(
        plt,
        log2.(Ns),
        [error[(N, scheme)] for N in Ns],
        seriestype = :scatter,
             shape = :star5,
        markersize = 6,
            xscale = :log10,
            yscale = :log10,
            xlabel = "log₂N",
             label = "L₁-norm, c(x) "*string(labels[scheme]),
            legend = :bottomleft,
             title = "Convergence Rates"
        )

end

for i in 1:length(schemes)
    
    scheme = schemes[i]
    
    plot!(
        plt,
        log2.(Ns[end-3:end]),
        [error[(N, scheme)] for N in Ns][end-3] .* (Ns[end-3] ./ Ns[end-3:end]) .^ roc[i],
        linestyle = :solid,
               lw = 3,
        label = raw"N^{-" * "$i" * raw"}" |> latexstring
    )
    
end

display(plt)

savefig(plt, "convergence_rates")

for i in 1:length(schemes)

    scheme = schemes[i] 
      name = labels[scheme]
    
    j = 3
    ROC = log10([error[(N, scheme)] for N in Ns][j-1] / [error[(N, scheme)] for N in Ns][j]) / log10(Ns[j-1] / Ns[j])
    println("Method = ", name, ", Rate of convergence = ", ROC, " expected = ", roc[i]) 
end

