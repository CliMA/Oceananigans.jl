# Wavenumber spectra of velocity fields: directional and isotropic spectra on doubly periodic grids, and zonal
# spectra along wet rows of a global grid. Wavenumbers are in index units, K ∈ [0, π].

using Statistics

one_sided_dft(N) = [exp(-2π * im * (k - 1) * (n - 1) / N) for k in 1:N÷2+1, n in 1:N]
full_dft(N) = [exp(-2π * im * (k - 1) * (n - 1) / N) for k in 1:N, n in 1:N]

index_wavenumbers(N) = [2π * (k - 1) / N for k in 1:N÷2+1]

hann(N) = [sin(π * (n - 1) / N)^2 for n in 1:N]

# One-sided power spectrum along the first dimension, averaged over the columns, normalized so that the sum over
# wavenumbers equals the mean square of the columns
function row_spectrum(columns; window=nothing)
    N = size(columns, 1)
    w = isnothing(window) ? ones(N) : window
    power = abs2.(one_sided_dft(N) * (w .* columns)) ./ sum(abs2, w) ./ N
    power[2:end-1, :] .*= 2
    iseven(N) || (power[end, :] .*= 2)
    return vec(mean(power, dims=2))
end

# Spectrum of `a` along x (dims = 1) or along y (dims = 2) on a doubly periodic grid
directional_spectrum(a, dims) = row_spectrum(dims == 1 ? a : permutedims(a))

# Isotropic spectrum of ½(u² + v²) on a doubly periodic N × N grid, in shells of width 2π/N
function isotropic_spectrum(u, v)
    N = size(u, 1)
    F = full_dft(N)
    density = (abs2.(F * u * transpose(F)) .+ abs2.(F * v * transpose(F))) ./ (2 * N^4)
    wavenumbers = [2π * min(k - 1, N - k + 1) / N for k in 1:N]
    shells = zeros(N ÷ 2 + 1)
    for j in 1:N, i in 1:N
        K = sqrt(wavenumbers[i]^2 + wavenumbers[j]^2)
        n = round(Int, K / (2π / N)) + 1
        n ≤ length(shells) && (shells[n] += density[i, j])
    end
    return index_wavenumbers(N), shells
end

# Zonal spectrum over the latitude rows whose points are all wet (exactly periodic in longitude)
function wet_row_spectrum(field, wet, rows)
    complete = [j for j in rows if all(wet[:, j])]
    isempty(complete) && return nothing
    return row_spectrum(field[:, complete])
end

# Zonal spectrum from Hann-windowed segments of `segment` points that are entirely wet, inside a longitude box
function box_spectrum(field, wet, rows, columns; segment=48)
    pieces = Vector{Float64}[]
    for j in rows
        for start in first(columns):segment÷2:last(columns)-segment+1
            window = start:start+segment-1
            all(wet[mod1.(window, size(field, 1)), j]) && push!(pieces, field[mod1.(window, size(field, 1)), j])
        end
    end
    isempty(pieces) && return nothing
    return row_spectrum(reduce(hcat, pieces); window=hann(segment)), length(pieces)
end
