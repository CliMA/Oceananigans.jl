using CairoMakie, JLD2, Statistics, HDF5, Oceananigans, ProgressBars
data_directory = "/nobackup1/sandre/OceananigansData/"


casevar = 6
# jlfile = jldopen("baroclinic_double_gyre_free_surface_$casevar.jld2", "r")
# jlfile2 = jldopen("baroclinic_double_gyre_$casevar.jld2", "r")
# ηkeys =  keys(jlfile["timeseries"]["η"])[2:end]

η = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_free_surface_$casevar.jld2", "η"; backend = InMemory(10))
u = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "u";              backend = InMemory(10))
v = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "v";              backend = InMemory(10))
w = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "w";              backend = InMemory(10))
b = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "b";              backend = InMemory(10))

η = zeros(size(jlfile["timeseries"]["η"]["0"])[1:2]..., length(ηkeys))
M, N, L = size(jlfile2["timeseries"]["b"]["0"])
u = zeros(M, N, L, length(ηkeys))
v = zeros(M, N, L, length(ηkeys))
w = zeros(M, N, L, length(ηkeys))
b = zeros(M, N, L, length(ηkeys))
for (i, ηkey) in enumerate(ηkeys)
    η[:, :, i] .= jlfile["timeseries"]["η"][ηkey]
    u[:, :, :, i] .= (jlfile2["timeseries"]["u"][ηkey][2:end, :, :] + jlfile2["timeseries"]["u"][ηkey][1:end-1, :, :])/2
    v[:, :, :, i] .= (jlfile2["timeseries"]["v"][ηkey][:, 2:end, :] + jlfile2["timeseries"]["v"][ηkey][:, 1:end-1, :])/2
    w[:, :, :, i] .= (jlfile2["timeseries"]["w"][ηkey][:, :, 2:end] + jlfile2["timeseries"]["w"][ηkey][:, :, 1:end-1])/2
    b[:, :, :, i] .= jlfile2["timeseries"]["b"][ηkey]
end
close(jlfile)
close(jlfile2)

Nt = length(u.times)
NN = 3
fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, interior(η[Nt - ii], :, :, 1), colormap = :viridis)
    end
end
save("etafield.png", fig)

fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, interior(η[ii], :, :, 1), colormap = :viridis)
    end
end
save("etafield_start.png", fig)

for k in 1:2
    fig = Figure() 
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
            heatmap!(ax, interior(u[Nt - ii], :, :, k), colormap = :viridis)
        end
    end
    save("ufield$k.png", fig)
end

for k in 1:2
    fig = Figure() 
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
            heatmap!(ax, interior(v[Nt - ii], :, :, k), colormap = :viridis)
        end
    end
    save("vfield$k.png", fig)
end

fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, interior(v[Nt - ii], :, :, 2), colormap = :viridis)
    end
end
save("vfield2.png", fig)


fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, interior(b[Nt - ii], :, :, 1), colormap = :viridis)
    end
end

save("bfield1.png", fig)

fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, interior(b[Nt - ii], :, :, 2), colormap = :viridis)
    end
end

save("bfield2.png", fig)

fig = Figure() 
for i in 1:NN
    for j in 1:NN
        ii = (i - 1) * NN + j
        ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y")
        heatmap!(ax, interior(w[Nt - ii], :, :, 2), colormap = :viridis)
    end
end

save("wfield2.png", fig)

squareheight = [mean(interior(η[i], :, :, 1) .^2) for i in eachindex(ηkeys)]

fig = Figure()
ax = Axis(fig[1,1]; xlabel = "time", ylabel = "mean(η^2)")
lines!(ax, squareheight[120:end], color = :blue)
save("squareheight.png", fig)

##

using Oceananigans.Utils
using Oceananigans.Architectures: device
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators
using Oceananigans.Architectures: architecture

function barotropic_streamfunction(u)
    U = Field(Integral(u, dims = 3))
    compute!(U)
    ψ = Field{Face, Face, Nothing}(u.grid)
    D = device(architecture(u.grid))

    _compute_ψ!(D, 16, u.grid.Nx+1)(ψ, u.grid, U)

    return ψ
end

@kernel function _compute_ψ!(ψ, grid, U)
    i = @index(Global, Linear)
    ψ[i, 1, 1] = 0

    for j in 2:grid.Ny
        ψ[i, j, 1] = ψ[i, j-1, 1] + U[i, j, 1] * Δyᶠᶜᶜ(i, j, 1, grid)
    end
end
ψ = barotropic_streamfunction(u[1000])

##
fig = Figure() 
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
heatmap!(ax, ψ, colormap = :viridis)
save("streamfunction.png", fig)
##
si = 120 #starting index
η̄  = Field{Center, Center, Nothing}(u.grid)
ση = Field{Center, Center, Nothing}(u.grid)
b̄  = CenterField(u.grid)
σb = CenterField(u.grid)
v̄  = XFaceField(u.grid)
σv = XFaceField(u.grid)
ū  = YFaceField(u.grid)
σu = YFaceField(u.grid)

averaging_steps = Nt - si + 1 : Nt
samples = length(averaging_steps)

for t in averaging_steps
    η̄ .+= η[t] / samples
    v̄ .+= v[t] / samples
    ū .+= u[t] / samples
    b̄ .+= b[t] / samples
end

for t in averaging_steps
    ση .+= (η[t] .- η̄) .^ 2 / samples
    σv .+= (v[t] .- v̄) .^ 2 / samples
    σu .+= (u[t] .- ū) .^ 2 / samples
    σb .+= (b[t] .- b̄) .^ 2 / samples
end

ση .= sqrt.(ση)
σv .= sqrt.(σv)
σu .= sqrt.(σu)
σb .= sqrt.(σb)

# hfile = h5open(data_directory * "baroclinic_double_gyre.hdf5", "w")
# hfile["eta"] = rη
# hfile["u"] = ru
# hfile["v"] = rv
# hfile["b"] = rb
# hfile["mean eta"] = η̄
# hfile["mean u"]  = ū
# hfile["mean v"]  = v̄
# hfile["mean b"]  = b̄
# hfile["std eta"] = ση
# hfile["std u"]   = σu
# hfile["std v"]   = σv
# hfile["std b"]   = σb
# close(hfile)

# state = zeros(M, N, 4, length(ηkeys) - si+1)
# state[:, :, 1, :] .= ru[:, :, 1, :]
# state[:, :, 2, :] .= rv[:, :, 1, :]
# state[:, :, 3, :] .= rb[:, :, 1, :]
# state[:, :, 4, :] .= rη[:, :, :]
# hfile = h5open(data_directory * "baroclinic_training_data.hdf5", "w")
# hfile["timeseries"] = state
# close(hfile)

using FFTW
using FFTW
using Oceananigans.Grids: φnode
using Statistics: mean

struct Spectrum{F, S}
    freq :: F
    spec :: S
end

import Base

Base.:(+)(s::Spectrum, t::Spectrum) = Spectrum(s.freq, s.spec .+ t.spec)
Base.:(*)(s::Spectrum, t::Spectrum) = Spectrum(s.freq, s.spec .* t.spec)
Base.:(/)(s::Spectrum, t::Int)      = Spectrum(s.frew, s.spec ./ t)

Base.real(s::Spectrum) = Spectrum(s.freq, real.(s.spec))
Base.abs(s::Spectrum)  = Spectrum(s.freq,  abs.(s.spec))

@inline onefunc(args...)  = 1.0
@inline hann_window(n, N) = sin(π * n / N)^2 

function power_cospectrum_1d(var1, var2, x; windowing = onefunc)

    Nx = length(x)
    Nfx = Int64(Nx)
    
    spectra = zeros(ComplexF64, Int(Nfx/2))
    
    dx = x[2] - x[1]

    freqs = fftfreq(Nfx, 1.0 / dx) # 0, +ve freq,-ve freqs (lowest to highest)
    freqs = freqs[1:Int(Nfx/2)] .* 2.0 .* π
    
    windowed_var1 = [var1[i] * windowing(i, Nfx) for i in 1:Nfx]
    windowed_var2 = [var2[i] * windowing(i, Nfx) for i in 1:Nfx]
    fourier1      = fft(windowed_var1) / Nfx
    fourier2      = fft(windowed_var2) / Nfx
    spectra[1]    += fourier1[1] .* conj(fourier2[1]) .+ fourier2[1] .* conj(fourier1[1])

    for m in 2:Int(Nfx/2)
        spectra[m] += fourier1[m] .* conj(fourier2[m]) .+ fourier2[m] .* conj(fourier1[m])
    end

    return Spectrum(freqs, spectra)
end

function zonal_spectrum(field, j, k; windowing = hann_window)
    x, y, z = nodes(field)

    var1 = Array(interior(field, :, j, k))
    var2 = Array(interior(field, :, j, k))

    return power_cospectrum_1d(var1, var2, x; windowing)
end


nval = Nt-101 

uNt = Field(@at((Center, Center, Center), u[nval]))
vNt = Field(@at((Center, Center, Center), v[nval]))

compute!(uNt)
compute!(vNt)

# An example: zonal Kinetic Energy spectrum at j = 128, k = 1 and time Nt
𝒰 = zonal_spectrum(uNt, 128, 1)
𝒱 = zonal_spectrum(vNt, 128, 1)

# Keep only the real part
E = real(𝒰 + 𝒱)

for nval in ProgressBar(Nt-100:Nt)
    uNt = Field(@at((Center, Center, Center), u[nval]))
    vNt = Field(@at((Center, Center, Center), v[nval]))

    compute!(uNt)
    compute!(vNt)

    # An example: zonal Kinetic Energy spectrum at j = 128, k = 1 and time Nt
    𝒰 = zonal_spectrum(uNt, 128, 1)
    𝒱 = zonal_spectrum(vNt, 128, 1)

    # Keep only the real part
    E += real(𝒰 + 𝒱)

end

fig = Figure()
ax  = Axis(fig[1, 1]; xlabel = "k", ylabel = "E(k)")
scatter!(log10.(E.freq[2:end]), log10.(abs.(E.spec[2:end])), color = :blue)
save("ke_spectrum_at_128_1.png", fig)
