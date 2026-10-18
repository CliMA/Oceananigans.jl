#####
##### Conservative remapping of one column between two partitions of a monotone integration coordinate,
##### using the FV3 subgrid reconstructions (Lin 2004; GFDL_atmos_cubed_sphere model/fv_operators.F90).
#####
##### Columns are indexed top-to-bottom (k = 1 is the top layer, interfaces p[1] < ... < p[Nz+1]), as in FV3.
##### Bottom-first storage passes reversed views; the arithmetic order is FV3's regardless of storage.
#####

"""
    FV3ScalarProfile(; kord, iv, minimum_value)

FV3 `scalar_profile` reconstruction (used by `map_scalar` and `map1_q2`). `iv = 0` enforces positivity,
`iv = 1` is for general scalars, `iv = -1` for winds, `iv = -2` uses the lower boundary value.
`minimum_value` is FV3's `q_min`, which affects only `|kord| ∈ (9, 10, 11)`.
"""
struct FV3ScalarProfile{FT}
    kord :: Int
    iv :: Int
    minimum_value :: FT

    function FV3ScalarProfile(kord, iv, minimum_value::FT) where FT
        validate_profile(kord, iv, (-2, -1, 0, 1, 2), "FV3ScalarProfile")
        return new{FT}(kord, iv, minimum_value)
    end
end

FV3ScalarProfile(; kord, iv, minimum_value) = FV3ScalarProfile(kord, iv, minimum_value)

# FV3 calls ppm_profile for kord ≤ 7 (not implemented here) and aborts for kord ∉ 8:15 ∪ {17}.
function validate_profile(kord, iv, supported_iv, name)
    abs(kord) ∈ (8, 9, 10, 11, 12, 13, 14, 15, 17) && kord > 7 ||
        throw(ArgumentError("$name: kord = $kord is not supported (FV3 selects these reconstructions only for kord ∈ 8:15 ∪ {17})"))
    iv ∈ supported_iv || throw(ArgumentError("$name: iv = $iv is not supported; expected one of $supported_iv"))
    return nothing
end

"""
    FV3CubicSplineProfile(; kord, iv)

FV3 `cs_profile` reconstruction (used by `map1_ppm`). `iv = -3` additionally imposes the lower boundary value.
"""
struct FV3CubicSplineProfile
    kord :: Int
    iv :: Int

    function FV3CubicSplineProfile(kord, iv)
        validate_profile(kord, iv, (-3, -2, -1, 0, 1, 2), "FV3CubicSplineProfile")
        return new(kord, iv)
    end
end

FV3CubicSplineProfile(; kord, iv) = FV3CubicSplineProfile(kord, iv)

const FV3Profile = Union{FV3ScalarProfile, FV3CubicSplineProfile}

minimum_value(p::FV3ScalarProfile) = p.minimum_value
minimum_value(::FV3CubicSplineProfile) = nothing

# The FV3 overlap search assumes increasing interfaces and target edges inside the source column; outside that
# domain FV3 returns partial or undefined sums, so such inputs are rejected before any arithmetic.
function validate_remap_domain(p₂, p₁, Nz₂, Nz₁, target_thickness)
    length(p₁) == Nz₁ + 1 || throw(ArgumentError("source interfaces must have length $(Nz₁ + 1), got $(length(p₁))"))
    length(p₂) == Nz₂ + 1 || throw(ArgumentError("target interfaces must have length $(Nz₂ + 1), got $(length(p₂))"))
    Nz₁ >= 3 || throw(ArgumentError("the FV3 reconstructions need at least 3 source layers, got $Nz₁"))
    all(isfinite, p₁) && all(isfinite, p₂) || throw(ArgumentError("interfaces must be finite"))
    all(p₁[k+1] > p₁[k] for k in 1:Nz₁) || throw(ArgumentError("source interfaces must be strictly increasing"))
    all(p₂[k+1] > p₂[k] for k in 1:Nz₂) || throw(ArgumentError("target interfaces must be strictly increasing"))
    p₂[1] >= p₁[1] && p₂[Nz₂+1] <= p₁[Nz₁+1] ||
        throw(ArgumentError("target column [$(p₂[1]), $(p₂[Nz₂+1])] is not inside the source column [$(p₁[1]), $(p₁[Nz₁+1])]"))
    isnothing(target_thickness) || length(target_thickness) == Nz₂ ||
        throw(ArgumentError("target_thickness must have length $Nz₂"))
    return nothing
end

"""
    FV3RemapWorkspace(FT, Nz)

Scratch storage for one column reconstruction with `Nz` layers.
"""
struct FV3RemapWorkspace{A, V, B}
    a4 :: A       # (mean, left edge, right edge, curvature) per layer
    q :: V        # interface values, length Nz + 1
    γ :: V        # tridiagonal coefficients, then layer-mean differences
    δp :: V       # source layer thickness
    extm :: B
    ext5 :: B
    ext6 :: B
end

FV3RemapWorkspace(FT, Nz) = FV3RemapWorkspace(zeros(FT, 4, Nz), zeros(FT, Nz + 1), zeros(FT, Nz + 1), zeros(FT, Nz),
                                              falses(Nz), falses(Nz), falses(Nz))

# gfortran MIN/MAX semantics: the first argument wins ties (including ±0), unlike Base.min/max.
@inline fortran_max(a, b) = ifelse((b > a) | isnan(a), b, a)
@inline fortran_min(a, b) = ifelse((b < a) | isnan(a), b, a)
@inline fortran_max(a, b, c) = fortran_max(fortran_max(a, b), c)
@inline fortran_min(a, b, c) = fortran_min(fortran_min(a, b), c)

#####
##### Limiters (cs_limiters)
#####

@inline function cs_limiter!(a4, k, extremum, mode)
    FT = eltype(a4)
    r12 = convert(FT, 1//12)
    a1, a2, a3, a6 = a4[1, k], a4[2, k], a4[3, k], a4[4, k]

    if mode == 0
        if a1 <= 0
            a4[2, k] = a1
            a4[3, k] = a1
            a4[4, k] = zero(FT)
        elseif abs(a3 - a2) < -a6
            if (a1 + convert(FT, 1//4) * (a3 - a2)^2 / a6 + a6 * r12) < 0
                if a1 < a3 && a1 < a2
                    a4[3, k] = a1
                    a4[2, k] = a1
                    a4[4, k] = zero(FT)
                elseif a3 > a2
                    a4[4, k] = 3 * (a2 - a1)
                    a4[3, k] = a2 - a4[4, k]
                else
                    a4[4, k] = 3 * (a3 - a1)
                    a4[2, k] = a3 - a4[4, k]
                end
            end
        end
    elseif mode == 1
        if (a1 - a2) * (a1 - a3) >= 0
            a4[2, k] = a1
            a4[3, k] = a1
            a4[4, k] = zero(FT)
        else
            da1 = a3 - a2
            da2 = da1^2
            a6da = a6 * da1
            if a6da < -da2
                a4[4, k] = 3 * (a2 - a1)
                a4[3, k] = a2 - a4[4, k]
            elseif a6da > da2
                a4[4, k] = 3 * (a3 - a1)
                a4[2, k] = a3 - a4[4, k]
            end
        end
    else
        if extremum
            a4[2, k] = a1
            a4[3, k] = a1
            a4[4, k] = zero(FT)
        else
            da1 = a3 - a2
            da2 = da1^2
            a6da = a6 * da1
            if a6da < -da2
                a4[4, k] = 3 * (a2 - a1)
                a4[3, k] = a2 - a4[4, k]
            elseif a6da > da2
                a4[4, k] = 3 * (a3 - a1)
                a4[2, k] = a3 - a4[4, k]
            end
        end
    end
    return nothing
end

@inline curvature(a4, k) = 3 * (2 * a4[1, k] - (a4[2, k] + a4[3, k]))
@inline curvature6(a4, k) = 6 * a4[1, k] - 3 * (a4[2, k] + a4[3, k])

# Huynh's second constraint on both edges of layer k.
@inline function huynh_edges!(a4, γ, k)
    FT = eltype(a4)
    pmp₁ = a4[1, k] - 2 * γ[k+1]
    lac₁ = pmp₁ + convert(FT, 3//2) * γ[k+2]
    a4[2, k] = fortran_min(fortran_max(a4[2, k], fortran_min(a4[1, k], pmp₁, lac₁)), fortran_max(a4[1, k], pmp₁, lac₁))
    pmp₂ = a4[1, k] + 2 * γ[k]
    lac₂ = pmp₂ - convert(FT, 3//2) * γ[k-1]
    a4[3, k] = fortran_min(fortran_max(a4[3, k], fortran_min(a4[1, k], pmp₂, lac₂)), fortran_max(a4[1, k], pmp₂, lac₂))
    return nothing
end

@inline function flatten!(a4, k)
    a4[2, k] = a4[1, k]
    a4[3, k] = a4[1, k]
    a4[4, k] = zero(eltype(a4))
    return nothing
end

#####
##### Interface values
#####

function interface_values!(w, qs, Nz, profile)
    a4, q, γ, δp = w.a4, w.q, w.γ, w.δp
    FT = eltype(a4)
    iv = profile.iv

    if iv == -2
        γ[2] = convert(FT, 1//2)
        q[1] = convert(FT, 3//2) * a4[1, 1]
        for k = 2:Nz-1
            grat = δp[k-1] / δp[k]
            bet = 2 + grat + grat - γ[k]
            q[k] = (3 * (a4[1, k-1] + a4[1, k]) - q[k-1]) / bet
            γ[k+1] = grat / bet
        end
        grat = δp[Nz-1] / δp[Nz]
        q[Nz] = (3 * (a4[1, Nz-1] + a4[1, Nz]) - grat * qs - q[Nz-1]) / (2 + grat + grat - γ[Nz])
        q[Nz+1] = qs
        for k = Nz-1:-1:1
            q[k] = q[k] - γ[k+1] * q[k+1]
        end
        return nothing
    end

    grat = δp[2] / δp[1]
    bet = grat * (grat + convert(FT, 1//2))
    q[1] = ((grat + grat) * (grat + 1) * a4[1, 1] + a4[1, 2]) / bet
    γ[1] = (1 + grat * (grat + convert(FT, 3//2))) / bet

    if iv == -3 && profile isa FV3CubicSplineProfile
        d4 = zero(FT)
        for k = 2:Nz-1
            d4 = δp[k-1] / δp[k]
            bet = 2 + d4 + d4 - γ[k-1]
            q[k] = (3 * (a4[1, k-1] + d4 * a4[1, k]) - q[k-1]) / bet
            γ[k] = d4 / bet
        end
        d4 = δp[Nz-1] / δp[Nz]
        bet = 2 + d4 + d4 - γ[Nz-1]
        grat = δp[Nz-1] / δp[Nz]
        # FV3 uses the loop's exit index k = Nz here (fv_operators.F90:991).
        q[Nz] = (3 * (a4[1, Nz-1] + d4 * a4[1, Nz]) - grat * qs - q[Nz-1]) / bet
        q[Nz+1] = qs
    else
        d4 = zero(FT)
        for k = 2:Nz
            d4 = δp[k-1] / δp[k]
            bet = 2 + d4 + d4 - γ[k-1]
            q[k] = (3 * (a4[1, k-1] + d4 * a4[1, k]) - q[k-1]) / bet
            γ[k] = d4 / bet
        end
        a_bot = 1 + d4 * (d4 + convert(FT, 3//2))
        q[Nz+1] = (2 * d4 * (d4 + 1) * a4[1, Nz] + a4[1, Nz-1] - a_bot * q[Nz]) / (d4 * (d4 + convert(FT, 1//2)) - a_bot * γ[Nz])
    end

    for k = Nz:-1:1
        q[k] = q[k] - γ[k] * q[k+1]
    end
    return nothing
end

#####
##### Reconstruction (scalar_profile / cs_profile)
#####

function reconstruct!(w, qs, Nz, profile::FV3Profile)
    a4, q, γ = w.a4, w.q, w.γ
    extm, ext5, ext6 = w.extm, w.ext5, w.ext6
    FT = eltype(a4)
    iv = profile.iv
    kord = abs(profile.kord)
    qmin = minimum_value(profile)

    interface_values!(w, qs, Nz, profile)

    if kord == 17
        for k = 1:Nz
            a4[2, k] = q[k]
            a4[3, k] = q[k+1]
            a4[4, k] = curvature(a4, k)
        end
        return nothing
    end

    q[2] = fortran_min(q[2], fortran_max(a4[1, 1], a4[1, 2]))
    q[2] = fortran_max(q[2], fortran_min(a4[1, 1], a4[1, 2]))

    for k = 2:Nz
        γ[k] = a4[1, k] - a4[1, k-1]
    end

    for k = 3:Nz-1
        if kord >= 14 || γ[k-1] * γ[k+1] > 0
            q[k] = fortran_min(q[k], fortran_max(a4[1, k-1], a4[1, k]))
            q[k] = fortran_max(q[k], fortran_min(a4[1, k-1], a4[1, k]))
        elseif γ[k-1] > 0
            q[k] = fortran_max(q[k], fortran_min(a4[1, k-1], a4[1, k]))
        else
            q[k] = fortran_min(q[k], fortran_max(a4[1, k-1], a4[1, k]))
            iv == 0 && (q[k] = fortran_max(zero(FT), q[k]))
        end
    end

    q[Nz] = fortran_min(q[Nz], fortran_max(a4[1, Nz-1], a4[1, Nz]))
    q[Nz] = fortran_max(q[Nz], fortran_min(a4[1, Nz-1], a4[1, Nz]))

    for k = 1:Nz
        a4[2, k] = q[k]
        a4[3, k] = q[k+1]
    end

    for k = 1:Nz
        if k == 1 || k == Nz
            extm[k] = (a4[2, k] - a4[1, k]) * (a4[3, k] - a4[1, k]) > 0
        else
            extm[k] = γ[k] * γ[k+1] < 0
        end
        if kord > 9
            x0 = 2 * a4[1, k] - (a4[2, k] + a4[3, k])
            x1 = abs(a4[2, k] - a4[3, k])
            a4[4, k] = 3 * x0
            ext5[k] = abs(x0) > x1
            ext6[k] = abs(a4[4, k]) > x1
        end
    end

    if iv == 0
        a4[2, 1] = fortran_max(zero(FT), a4[2, 1])
    elseif iv == -1
        a4[2, 1] * a4[1, 1] <= 0 && (a4[2, 1] = zero(FT))
    elseif iv == 2
        flatten!(a4, 1)
    end

    if iv != 2
        a4[4, 1] = curvature(a4, 1)
        cs_limiter!(a4, 1, extm[1], 1)
    end

    a4[4, 2] = curvature(a4, 2)
    cs_limiter!(a4, 2, extm[2], 2)

    scalar = profile isa FV3ScalarProfile

    for k = 3:Nz-2
        if kord <= 8
            huynh_edges!(a4, γ, k)
            a4[4, k] = curvature(a4, k)
        elseif kord == 9
            if extm[k] && extm[k-1]
                flatten!(a4, k)
            elseif extm[k] && extm[k+1]
                flatten!(a4, k)
            elseif scalar && extm[k] && a4[1, k] < qmin
                flatten!(a4, k)
            else
                a4[4, k] = scalar ? curvature(a4, k) : curvature6(a4, k)
                if abs(a4[4, k]) > abs(a4[2, k] - a4[3, k])
                    huynh_edges!(a4, γ, k)
                    a4[4, k] = scalar ? curvature(a4, k) : curvature6(a4, k)
                end
            end
        elseif kord == 10
            if extm[k]
                if (scalar && a4[1, k] < qmin) || extm[k-1] || extm[k+1]
                    flatten!(a4, k)
                else
                    a4[4, k] = curvature6(a4, k)
                end
            else
                a4[4, k] = curvature6(a4, k)
                if abs(a4[4, k]) > abs(a4[2, k] - a4[3, k])
                    huynh_edges!(a4, γ, k)
                    a4[4, k] = curvature6(a4, k)
                end
            end
        elseif kord == 11
            if ext5[k] && (ext5[k-1] || ext5[k+1] || (scalar && a4[1, k] < qmin))
                flatten!(a4, k)
            else
                a4[4, k] = curvature(a4, k)
            end
        elseif kord == 12
            if ext5[k]
                if ext5[k-1] || ext5[k+1]
                    a4[2, k] = a4[1, k]
                    a4[3, k] = a4[1, k]
                elseif ext6[k-1] || ext6[k+1]
                    huynh_edges!(a4, γ, k)
                end
            elseif ext6[k]
                (ext5[k-1] || ext5[k+1]) && huynh_edges!(a4, γ, k)
            end
            a4[4, k] = curvature(a4, k)
        elseif kord == 13
            a4[4, k] = curvature(a4, k)
        elseif kord == 14
            cs_limiter!(a4, k, extm[k], 2)
        elseif kord == 15
            cs_limiter!(a4, k, extm[k], 1)
        end

        iv == 0 && kord <= 13 && cs_limiter!(a4, k, extm[k], 0)
    end

    if iv == 0
        a4[3, Nz] = fortran_max(zero(FT), a4[3, Nz])
    elseif iv == -1
        a4[3, Nz] * a4[1, Nz] <= 0 && (a4[3, Nz] = zero(FT))
    end

    for k = Nz-1:Nz
        a4[4, k] = curvature(a4, k)
        k == Nz - 1 && cs_limiter!(a4, k, extm[k], 2)
        k == Nz && cs_limiter!(a4, k, extm[k], 1)
    end

    return nothing
end

#####
##### Overlap integration (map_scalar / map1_ppm / map1_q2)
#####

"""
    remap_column!(q₂, p₂, q₁, p₁, qs, profile, workspace; target_thickness = nothing)

Remap the layer means `q₁` on source interfaces `p₁` (length `Nz₁ + 1`, increasing) to layer means `q₂` on target
interfaces `p₂`, conserving `∑ q Δp`. `qs` is the lower boundary value (used only by `iv = -2, -3`). The content
of each target layer is divided by `p₂[k+1] - p₂[k]` (FV3 `map_scalar`, `map1_ppm`) or, if given, by
`target_thickness[k]` (FV3 `map1_q2`). Interfaces must increase strictly and the target column must lie within the
source column; other inputs throw `ArgumentError` before any arithmetic (FV3 returns partial or undefined sums there).
"""
function remap_column!(q₂, p₂, q₁, p₁, qs, profile::FV3Profile, w::FV3RemapWorkspace; target_thickness = nothing)
    FT = eltype(q₂)
    r3 = convert(FT, 1//3)
    r23 = convert(FT, 2//3)
    Nz₁ = length(q₁)
    Nz₂ = length(q₂)
    a4, δp = w.a4, w.δp
    validate_remap_domain(p₂, p₁, Nz₂, Nz₁, target_thickness)
    length(δp) >= Nz₁ || throw(ArgumentError("workspace holds $(length(δp)) layers, need $Nz₁"))

    for k = 1:Nz₁
        δp[k] = p₁[k+1] - p₁[k]
        a4[1, k] = q₁[k]
    end

    reconstruct!(w, qs, Nz₁, profile)

    k₀ = 1
    qsum = convert(FT, NaN) # FV3 leaves qsum undefined if a target edge lies outside the source column
    for k = 1:Nz₂
        found = false
        complete = false
        for l = k₀:Nz₁
            if p₂[k] >= p₁[l] && p₂[k] <= p₁[l+1]
                found = true
                pl = (p₂[k] - p₁[l]) / δp[l]
                if p₂[k+1] <= p₁[l+1]
                    pr = (p₂[k+1] - p₁[l]) / δp[l]
                    q₂[k] = a4[2, l] + convert(FT, 1//2) * (a4[4, l] + a4[3, l] - a4[2, l]) * (pr + pl) -
                            a4[4, l] * r3 * (pr * (pr + pl) + pl^2)
                    k₀ = l
                    complete = true
                else
                    qsum = (p₁[l+1] - p₂[k]) * (a4[2, l] + convert(FT, 1//2) * (a4[4, l] +
                            a4[3, l] - a4[2, l]) * (1 + pl) - a4[4, l] *
                            (r3 * (1 + pl * (1 + pl))))
                    for m = l+1:Nz₁
                        if p₂[k+1] > p₁[m+1]
                            qsum = qsum + δp[m] * a4[1, m]
                        else
                            δ = p₂[k+1] - p₁[m]
                            esl = δ / δp[m]
                            qsum = qsum + δ * (a4[2, m] + convert(FT, 1//2) * esl *
                                   (a4[3, m] - a4[2, m] + a4[4, m] * (1 - r23 * esl)))
                            k₀ = m
                            break
                        end
                    end
                end
                break
            end
        end
        if !complete
            denominator = isnothing(target_thickness) ? p₂[k+1] - p₂[k] : target_thickness[k]
            q₂[k] = qsum / denominator
        end
    end

    return q₂
end
