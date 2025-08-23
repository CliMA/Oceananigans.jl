using CubedSphere
using LinearAlgebra
using Statistics
using Random

using Oceananigans.Grids: spherical_area_triangle, spherical_area_quadrilateral

"""
    spherical_distance(a₁, a₂)

Compute the great-circle arc length (angle in radians) between two points on the **unit sphere**, given their Cartesian
coordinates `a₁` and `a₂`. Both inputs are expected to be 3-vectors of unit length; the dot product is clamped to
`[-1, 1]` to guard against floating-point roundoff before applying `acos`.

# Arguments
- `a₁`, `a₂`: 3-element Cartesian vectors on the unit sphere (‖`a`‖ = 1).

# Returns
- The arc length (in radians) between `a₁` and `a₂`.
"""
function spherical_distance(a₁::AbstractVector, a₂::AbstractVector)
    (sum(a₁.^2) ≈ 1 && sum(a₂.^2) ≈ 1) || error("a₁ and a₂ must be unit vectors")

    # Compute the dot product and calculate the arccosine to find the angle.
    cosθ = dot(a₁, a₂)

    # Ensure the result is within the domain of acos due to potential floating-point errors.
    cosθ = clamp(cosθ, -1, 1)

    # Return the arc length, which is the angle between the two points.
    return acos(cosθ)
end

"""
    spherical_quadrilateral_vertices(X, Y, Z, i, j)

Return the four Cartesian vertex vectors of the spherical grid cell whose corners are indexed by `(i, j)`, `(i+1, j)`,
`(i+1, j+1)`, and `(i, j+1)` in the arrays `X`, `Y`, and `Z`. Each of `X`, `Y`, and `Z` is a 2D array of size `(Nx, Ny)`
holding the Cartesian coordinates of grid vertices on the sphere, such that the point at `(i, j)` is
`(X[i, j], Y[i, j], Z[i, j])`.

# Arguments
- `X`, `Y`, `Z`: `(Nx, Ny)` arrays of Cartesian coordinates on the sphere.
- `i`, `j`: Indices of the lower-left corner of the cell (in array order).

# Returns
- `(a₁, a₂, a₃, a₄)`: The four 3-element Cartesian vertex vectors at `(i, j)`, `(i+1, j)`, `(i+1, j+1)`, and `(i, j+1)`.
"""
function spherical_quadrilateral_vertices(X, Y, Z, i, j)
    x₁ = X[i, j]
    y₁ = Y[i, j]
    z₁ = Z[i, j]
    a₁ = [x₁, y₁, z₁]
    x₂ = X[i+1, j]
    y₂ = Y[i+1, j]
    z₂ = Z[i+1, j]
    a₂ = [x₂, y₂, z₂]
    x₃ = X[i+1, j+1]
    y₃ = Y[i+1, j+1]
    z₃ = Z[i+1, j+1]
    a₃ = [x₃, y₃, z₃]
    x₄ = X[i, j+1]
    y₄ = Y[i, j+1]
    z₄ = Z[i, j+1]
    a₄ = [x₄, y₄, z₄]

    return a₁, a₂, a₃, a₄
end

"""
    compute_deviation_from_isotropy(X, Y, Z)

Compute a scalar measure of the deviation from isotropy for a spherical grid (e.g., a conformal cubed-sphere panel),
defined by the coordinate arrays `X`, `Y`, and `Z`. Each of `X`, `Y`, and `Z` is a 2D array of size `(Nx, Ny)` holding
the Cartesian coordinates of grid vertices on the sphere, such that the point at `(i, j)` is
`(X[i, j], Y[i, j], Z[i, j])`. The grid therefore contains `(Nx-1) × (Ny-1)` spherical quadrilateral cells.

For each quadrilateral cell in the grid, the function computes the arc lengths of its four edges on the unit sphere and
evaluates the sum of consecutive edge differences as a measure of anisotropy. The total deviation is then returned as 
the Euclidean norm over all cells.

# Arguments
- `X`, `Y`, `Z`: `(Nx, Ny)` arrays of Cartesian coordinates on the sphere.

# Returns
- A nonnegative scalar quantifying overall grid anisotropy (larger ⇒ more anisotropic).
"""
function compute_deviation_from_isotropy(X, Y, Z)
    Nx, Ny = size(X)
    deviation_from_isotropy = zeros(Nx-1, Ny-1)

    for j in 1:Ny-1, i in 1:Nx-1
        a₁, a₂, a₃, a₄ = spherical_quadrilateral_vertices(X, Y, Z, i, j)

        # Compute the arc lengths (distances) between the points a₁ and a₂, a₂ and a₃, a₃ and a₄, and a₄ and a₁ on the
        # unit sphere.
        d₁ = spherical_distance(a₁, a₂)
        d₂ = spherical_distance(a₂, a₃)
        d₃ = spherical_distance(a₃, a₄)
        d₄ = spherical_distance(a₄, a₁)

        # Compute the deviation from isotropy.
        deviation_from_isotropy[i, j] = abs(d₁ - d₂) + abs(d₂ - d₃) + abs(d₃ - d₄) + abs(d₄ - d₁)
    end

    return norm(deviation_from_isotropy)
end

"""
    compute_cell_areas(X, Y, Z)

Compute the spherical surface areas of all quadrilateral cells in a spherical grid (e.g., a conformal cubed sphere 
panel) defined by the coordinate arrays `X`, `Y`, and `Z`. Each of `X`, `Y`, and `Z` is a 2D array of size `(Nx, Ny)` 
holding the Cartesian coordinates of the grid vertices on the sphere, such that the point at `(i, j)` is
`(X[i, j], Y[i, j], Z[i, j])`. The grid therefore contains `(Nx-1) × (Ny-1)` spherical quadrilateral cells.

For each cell centered at `(i, j)` with vertices `(i, j)`, `(i+1, j)`, `(i+1, j+1)`, and `(i, j+1)`, the function
computes the cell area using `spherical_area_quadrilateral` and stores the results in a 2D array.

# Arguments
- `X`, `Y`, `Z`: `(Nx, Ny)` arrays of Cartesian coordinates on the sphere.

# Returns
- `cell_areas`: An `(Nx-1, Ny-1)` array of spherical quadrilateral cell areas.
"""
function compute_cell_areas(X, Y, Z)
    Nx, Ny = size(X)
    cell_areas = zeros(Nx-1, Ny-1)

    for j in 1:Ny-1, i in 1:Nx-1
        a₁, a₂, a₃, a₄ = spherical_quadrilateral_vertices(X, Y, Z, i, j)
        cell_areas[i, j] = spherical_area_quadrilateral(a₁, a₂, a₃, a₄)
    end

    return cell_areas
end

"""
    geometric_spacing(N, ratio_raised_to_N_minus_one)

Construct a symmetric set of `N` face locations on the interval `[-1, 1]` with geometrically graded spacing away from
the domain center. Let `r` be the geometric ratio recovered from `ratio_raised_to_N_minus_one` via
`r = (ratio_raised_to_N_minus_one)^(1/(N-1))`. The gaps between consecutive faces grow by a factor of `r` as one moves
outward from the center, and the layout is mirrored about zero. Endpoints are fixed at `-1` and `+1`.

- **Odd `N`**: A face lies at `0`. Outward gaps are `Δx, Δx*r, Δx*r^2, …`, mirrored about `0`, and chosen so the
  rightmost face lands at `+1`.
- **Even `N`**: No face at `0`. Central faces are at `±Δx/2`; outward gaps are `Δx*r, Δx*r^2, …`, mirrored, and chosen
  so the rightmost face lands at `+1`.

# Arguments
- `N`: Number of faces (`N ≥ 2`).
- `ratio_raised_to_N_minus_one`: The value `r^(N-1)` for some geometric ratio `r > 0`. Values near `1` yield nearly 
  uniform spacing. (Exactly `1` is not supported by the closed-form formulas used here.)

# Returns
- `x_faces`: A length-`N` monotonically increasing vector of face coordinates on `[-1, 1]` with geometric grading and 
  symmetry: `x_faces[1] = -1`, `x_faces[N] = 1`, and `x_faces[i] = -x_faces[N+1-i]`.
"""
function geometric_spacing(N, ratio_raised_to_N_minus_one)
    ratio = ratio_raised_to_N_minus_one^(1/(N - 1))
    x_faces = zeros(N)

    if isodd(N)
        M = round(Int, (N + 1)/2)
    
        Δx = 1 * (ratio - 1) / (ratio^(M - 1) - 1)

        x_faces[M] = 0
        
        k = 0
        
        for i in M+1:N
            x_faces[i] = x_faces[i-1] + Δx * ratio^k
            x_faces[N+1-i] = -x_faces[i]
            k += 1
        end
        
        x_faces[1] = -1
        x_faces[N] = 1
    else
        M = Int(N/2)
    
        Δx = 1/((ratio^M - 1)/(ratio - 1) - 0.5)
        
        x_faces[M] = -0.5Δx
        x_faces[M+1] = 0.5Δx
        
        k = 1
        
        for i in M+2:N
            x_faces[i] = x_faces[i-1] + Δx * ratio^k
            x_faces[N+1-i] = -x_faces[i]
            k += 1
        end
        
        x_faces[1] = -1
        x_faces[N] = 1
    end 
    
    return x_faces
end

"""
    exponential_spacing(N, k₀ByN)

Construct a symmetric set of `N` face locations on `[-1, 1]` with **exponentially graded** spacing away from the domain
center. Let `k₀ = k₀ByN * N`, and define an exponential map on the right half, `x(t) = α·exp(t/k₀) + β`, anchored so 
that it passes through `(t₀, 0)` and `(t₁, 1)`, then mirror about zero. Endpoints are fixed at `-1` and `+1`.

- **Odd `N`** (`M = (N+1)/2`): a face lies at `0` (`x_faces[M] = 0`). The right-half faces use `t = 1, …, M` with 
  `x(1) = 0`, `x(M) = 1`, and the left half is the negative mirror.
- **Even `N`** (`M = N/2`): no face at `0`. The two central faces straddle zero, with `0` midway between them; the
  right-half faces use `t = 2, …, M+1` anchored by `x(1.5) = 0`, `x(M+1) = 1`, and the left half is mirrored.

# Arguments
- `N`: Number of faces (`N ≥ 2`).
- `k₀ByN`: Grading parameter scaled by `N` (the code uses `k₀ = k₀ByN * N`). Larger `k₀ByN` yields spacing closer to 
  uniform; smaller `k₀ByN` increases clustering near the center. Requires `k₀ByN > 0`.

# Returns
- `x_faces`: A length-`N` strictly increasing vector of face coordinates on `[-1, 1]` with symmetry 
  `x_faces[i] = -x_faces[N+1-i]`, and endpoints `x_faces[1] = -1`, `x_faces[N] = 1`.
"""
function exponential_spacing(N, k₀ByN)
    k₀ = k₀ByN * N
    x_faces = zeros(N)
    
    if isodd(N)
        M = round(Int, (N + 1)/2)
        
        A = [exp(1/k₀) 1
             exp(M/k₀) 1]

        b = [0, 1]
        
        coefficients = A \ b
        
        x_faces[M:N] = coefficients[1] * exp.((1:M)/k₀) .+ coefficients[2]
        
        for i in 1:M-1
            x_faces[i] = -x_faces[N+1-i]
        end
        
        x_faces[1] = -1
        x_faces[M] = 0
        x_faces[N] = 1
    else
        M = Int(N/2)
        
        A = [exp(1.5/k₀)   1
             exp((M+1)/k₀) 1]
    
        b = [0, 1]
        
        coefficients = A \ b
        
        x_faces[M+1:N] = coefficients[1] * exp.((2:M+1)/k₀) .+ coefficients[2]
        
        for i in 1:M
            x_faces[i] = -x_faces[N+1-i]
        end
        
        x_faces[1] = -1
        x_faces[N] = 1
    end

    return x_faces
end

"""
    conformal_cubed_sphere_coordinates(Nx, Ny;
                                       non_uniform_spacing=false,
                                       spacing_type="geometric",
                                       ratio_raised_to_Nx_minus_one=10.5,
                                       k₀ByNx=0.45)

Generate computational-space coordinates `x` and `y` on `[-1, 1] × [-1, 1]` and map them to Cartesian coordinates 
`(X, Y, Z)` on the sphere using `conformal_cubed_sphere_mapping`. The arrays `X`, `Y`, and `Z` are of size `(Nx, Ny)` 
and correspond to `Nx × Ny` grid vertices, defining `(Nx-1) × (Ny-1)` spherical quadrilateral cells of a conformal cubed
sphere panel.

If `non_uniform_spacing == false`, `x` and `y` have uniform spacing (equal increments), so their tensor product defines
a uniform grid on `[-1, 1] × [-1, 1]`. If `true`, symmetric graded spacing is applied on both axes:
- `spacing_type == "geometric"`: uses `geometric_spacing(N, ratio_raised_to_Nx_minus_one)` for each axis.
- `spacing_type == "exponential"`: uses `exponential_spacing(N, k₀ByNx)` for each axis.

# Arguments
- `Nx, Ny`: Number of grid vertices along the `x` and `y` directions (≥ 2).
- `non_uniform_spacing`: Enable graded (non-uniform) spacing on both axes.
- `spacing_type`: Either `"geometric"` or `"exponential"` (used only when `non_uniform_spacing` is true).
- `ratio_raised_to_Nx_minus_one`: Geometric grading control, interpreted as `r^(Nx-1)`.
- `k₀ByNx`: Exponential grading control used as `k₀ = k₀ByNx * Nx`.

# Returns
- `x`, `y`: Computational-space vertex coordinates of lengths `Nx` and `Ny`.
- `X`, `Y`, `Z`: `(Nx, Ny)` arrays of Cartesian coordinates on the sphere, with
  `X[i, j], Y[i, j], Z[i, j] = conformal_cubed_sphere_mapping(x[i], y[j])`.
"""
function conformal_cubed_sphere_coordinates(Nx, Ny;
                                            non_uniform_spacing = false,
                                            spacing_type = "geometric",
                                            ratio_raised_to_Nx_minus_one = 10.5,
                                            k₀ByNx = 0.45)
    x = range(-1, 1, length = Nx)
    y = range(-1, 1, length = Ny)

    if non_uniform_spacing
        if spacing_type == "geometric"
            # For Nx = Ny = 32 + 1, setting ratio = 1.0775 increases the minimum cell width by a factor of 1.92.
            # For Nx = Ny = 1024 + 1, setting ratio = 1.0042 increases the minimum cell width by a factor of 3.25.
            x = geometric_spacing(Nx, ratio_raised_to_Nx_minus_one)
            y = geometric_spacing(Ny, ratio_raised_to_Nx_minus_one)
        elseif spacing_type == "exponential"
            # For Nx = Ny = 32 + 1, setting k₀ByNx = 15 increases the minimum cell width by a factor of 1.84.
            # For Nx = Ny = 1024 + 1, setting k₀ByNx = 10 increases the minimum cell width by a factor of 2.58.
            x = exponential_spacing(Nx, k₀ByNx)
            y = exponential_spacing(Ny, k₀ByNx)
        end
    end
    
    X = zeros(length(x), length(y))
    Y = zeros(length(x), length(y))
    Z = zeros(length(x), length(y))
    
    for (j, y′) in enumerate(y), (i, x′) in enumerate(x)
        X[i, j], Y[i, j], Z[i, j] = conformal_cubed_sphere_mapping(x′, y′)
    end
    
    return x, y, X, Y, Z
end

"""
    specify_parameters(spacing_type)

Return a vector containing the initial guess for the spacing parameters used to build non-uniform conformal cubed sphere
panels. The parameterization depends on `spacing_type`:

- `"geometric"`: uses a single parameter interpreted downstream as `ratio^(N-1)` for geometric grading.
- `"exponential"`: uses a single parameter interpreted downstream as `k₀/N` for exponential grading.

Returns a one-element vector `[θ]` (a single parameter in this implementation).
"""
function specify_parameters(spacing_type)
    θ = 0
    if spacing_type == "geometric"
        ratio_raised_to_N = 1.0775
        θ = ratio_raised_to_N
    elseif spacing_type == "exponential"
        k₀ByN = 15
        θ = k₀ByN
    end
    return [θ]
end

"""
    specify_parameter_limits(spacing_type)

Return the lower and upper bounds for the single spacing parameter as a **2-element vector** `[min, max]`. For a
consistent interface with possible multi-parameter extensions, this vector is returned wrapped in a one-element array:
`[[min, max]]`. The bounds depend on `spacing_type`:

- `"geometric"`: returns `[[min, max]]` for `ratio^(N-1)`.
- `"exponential"`: returns `[[min, max]]` for `k₀/N`.

Returns `[θ_limits]` where `θ_limits == [min, max]` for the single parameter in this implementation.
"""
function specify_parameter_limits(spacing_type)
    θ_limits = zeros(2)
    if spacing_type == "geometric"
        ratio_raised_to_N_limits = [5, 15]
        θ_limits[1] = ratio_raised_to_N_limits[1]
        θ_limits[2] = ratio_raised_to_N_limits[2]
    elseif spacing_type == "exponential"
        k₀ByN_limits = [0.4, 0.5]
        θ_limits[1] = k₀ByN_limits[1]
        θ_limits[2] = k₀ByN_limits[2]
    end
    return [θ_limits]
end

"""
    specify_random_parameters(nEnsemble, spacing_type)

Draw an ensemble of random parameter vectors within the limits from `specify_parameter_limits(spacing_type)`. Each
ensemble member is sampled uniformly within its parameter bounds.

# Arguments
- `nEnsemble`: Number of ensemble members to generate.
- `spacing_type`: `"geometric"` or `"exponential"`.

# Returns
- A vector of length `nEnsemble`, where each element is a one-element parameter vector `[θ]` (a single parameter in this 
  implementation).
"""
function specify_random_parameters(nEnsemble, spacing_type)
    θ = specify_parameters(spacing_type)
    θ_limits = specify_parameter_limits(spacing_type)

    θᵣ = [[θ_limits[j][1] + (θ_limits[j][2] - θ_limits[j][1]) * rand() for j in 1:lastindex(θ)] for i in 1:nEnsemble]

    return θᵣ
end

"""
    specify_weights_for_model_diagnostics()

Return the weights applied to the model diagnostics used by the objective function. The two diagnostics are, in order: 
`(1) normalized minimum cell width`, `(2) deviation from isotropy`.

# Returns
- A 2-element vector of weights, e.g., `[10, 1]`.
"""
function specify_weights_for_model_diagnostics()
    weights = [10, 1]
    return weights
end

"""
    compute_model_diagnostics(X, Y, Z, minimum_reference_cell_area)

Compute the two model diagnostics for a given mapped grid:

1. **Normalized minimum cell width**: `sqrt(min(cell_area) / minimum_reference_cell_area)`,
   using spherical cell areas from `compute_cell_areas(X, Y, Z)` and a uniform-grid reference area.
2. **Deviation from isotropy**: the heuristic anisotropy measure from `compute_deviation_from_isotropy(X, Y, Z)`.

# Arguments
- `X, Y, Z`: `(Nx, Ny)` Cartesian coordinate arrays on the sphere.
- `minimum_reference_cell_area`: The minimum cell area of a reference (typically uniform) grid.

# Returns
- A 2-element vector `[normalized_minimum_cell_width, deviation_from_isotropy]`.
"""
function compute_model_diagnostics(X, Y, Z, minimum_reference_cell_area)
    cell_areas = compute_cell_areas(X, Y, Z)
    normalized_minimum_cell_width = sqrt(minimum(cell_areas)/minimum_reference_cell_area)
    
    deviation_from_isotropy = compute_deviation_from_isotropy(X, Y, Z)

    model_diagnostics = vcat(normalized_minimum_cell_width, deviation_from_isotropy)
    
    return model_diagnostics
end

"""
    compute_weighted_model_diagnostics(model_diagnostics)

Apply weights from `specify_weights_for_model_diagnostics()` to the unweighted diagnostics.

# Arguments
- `model_diagnostics`: A 2-element vector `[normalized_minimum_cell_width, deviation_from_isotropy]`.

# Returns
- A 2-element vector with weights applied, in the same order as the inputs.
"""
function compute_weighted_model_diagnostics(model_diagnostics)
    normalized_minimum_cell_width = model_diagnostics[1]
    deviation_from_isotropy = model_diagnostics[2]

    weights = specify_weights_for_model_diagnostics()

    weighted_model_diagnostics = vcat(weights[1] * normalized_minimum_cell_width, weights[2] * deviation_from_isotropy)

    return weighted_model_diagnostics
end

"""
    forward_map(Nx, Ny, spacing_type, θ)

Evaluate the (weighted) model diagnostics for a non-uniform conformal cubed-sphere panel defined by parameters `θ`. The
steps are:
1. Clamp `θ` to parameter limits from `specify_parameter_limits(spacing_type)`.
2. Build a **reference** uniform grid via `conformal_cubed_sphere_coordinates(Nx, Ny)` and compute
   `minimum_reference_cell_area`.
3. Build the **non-uniform** grid via `conformal_cubed_sphere_coordinates(Nx, Ny; non_uniform_spacing=true, ...)`,
   passing the parameters in `θ` according to `spacing_type`.
4. Compute model diagnostics and then apply weights.

# Arguments
- `Nx, Ny`: Number of grid vertices along the panel coordinates.
- `spacing_type`: `"geometric"` (interprets `θ[1]` as `ratio^(Nx-1)`) or `"exponential"` (interprets `θ[1]` as `k₀/Nx`).
- `θ`: Parameter vector (one element in this implementation).

# Returns
- A 2-element vector of **weighted** model diagnostics.
"""
function forward_map(Nx, Ny, spacing_type, θ)
    θ_limits = specify_parameter_limits(spacing_type)

    for i in 1:lastindex(θ)
        θ[i] = clamp(θ[i], θ_limits[i][1], θ_limits[i][2])
    end
    
    x_reference, y_reference, X_reference, Y_reference, Z_reference = conformal_cubed_sphere_coordinates(Nx, Ny)
    cell_areas = compute_cell_areas(X_reference, Y_reference, Z_reference)
    minimum_reference_cell_area = minimum(cell_areas)
    
    x, y, X, Y, Z = (
    conformal_cubed_sphere_coordinates(Nx, Ny; non_uniform_spacing = true, spacing_type = spacing_type,
                                       ratio_raised_to_Nx_minus_one = θ[1], k₀ByNx = θ[1]))
    
    model_diagnostics = compute_model_diagnostics(X, Y, Z, minimum_reference_cell_area)

    weighted_model_diagnostics = compute_weighted_model_diagnostics(model_diagnostics)

	return weighted_model_diagnostics
end

"""
    specify_ideal_weighted_model_diagnostics()

Return the target (ideal) values for the **weighted** model diagnostics used by the inversion. By default, the target
normalized minimum cell width is `4` and the target deviation from isotropy is `0`, then weights are applied in the same
order.

# Returns
- A 2-element vector of target **weighted** diagnostics.
"""
function specify_ideal_weighted_model_diagnostics()
    normalized_minimum_cell_width = 4

    deviation_from_isotropy = 0

    weights = specify_weights_for_model_diagnostics()

    ideal_weighted_model_diagnostics = vcat(weights[1] * normalized_minimum_cell_width,
                                            weights[2] * deviation_from_isotropy)

    return ideal_weighted_model_diagnostics
end

"""
    optimize!(Nx, Ny, spacing_type, θ; nIterations=10, Δt=1)

Run an Ensemble Kalman Inversion (EKI) to tune the spacing parameter(s) for a non-uniform conformal cubed sphere panel.
An ensemble of parameter vectors `θ` is iteratively updated so that the **weighted** model diagnostics produced by
`forward_map` match the ideal targets from `specify_ideal_weighted_model_diagnostics()`.

At each iteration:
1. **Forward evaluations (parallelized):** For each ensemble member `θ[n]`, compute the predicted diagnostics
   `G[n] = forward_map(Nx, Ny, spacing_type, θ[n])`.
2. **Ensemble statistics:** Form means `θ̄ = mean(θ)` and `G̅ = mean(G)`, then compute
   - Cross-covariance `Cᵘᵖ = cov(θ, G)` (shape: nθ × ndata), and
   - Data covariance `Cᵖᵖ = cov(G, G)` (shape: ndata × ndata),
   using the unbiased `(nEnsemble-1)` denominator.
3. **Perturbed observations:** For each member, build `y[n] = ideal + Δt * η[n]` where `η[n] ~ N(0, I)`. Here `Δt` sets 
   the **perturbation magnitude** of the observations.
4. **Residuals:** `r[n] = y[n] - G[n]`.
5. **Implicit update (Kalman-like step):** Update each parameter vector via θ[n] ← θ[n] + K * r[n], with 
   K = Cᵘᵖ * (Cᵖᵖ + I/Δt)⁻¹, implemented by solving the linear system with a Cholesky factorization of `Cᵖᵖ + I/Δt`. The
   same `Δt` also acts as an **implicit damping/step-size control**: smaller `Δt` ⇒ stronger regularization and smaller 
   updates; larger `Δt` ⇒ weaker regularization and larger, noisier updates.
6. **Monitoring:** Report `error = ‖mean(r)‖` and store a snapshot of the ensemble.

This function **mutates** the input ensemble `θ` in place (it becomes the final ensemble) and records the full ensemble
after each iteration.

# Arguments
- `Nx, Ny`: Number of grid vertices along the panel coordinates.
- `spacing_type`: `"geometric"` or `"exponential"`.
- `θ`: Initial ensemble — a vector of one-element parameter vectors `[θ]` (length `nEnsemble`).
- `nIterations`: Number of EKI iterations.
- `Δt`: Pseudo-time step that sets both the observation perturbation scale and the implicit damping
        in the linear solve `(Cᵖᵖ + I/Δt)`.

# Returns
- `θ_series`: A length `nIterations + 1` vector; each entry is a **snapshot** of the full ensemble (index 1 is the 
  initial ensemble; the last is the final ensemble). The input `θ` is also mutated to the final state.
"""
function optimize!(Nx, Ny, spacing_type, θ; nIterations = 10, Δt = 1)
    ideal_data = specify_ideal_weighted_model_diagnostics()
    model_data = forward_map(Nx, Ny, spacing_type, mean(θ))

    nData = length(ideal_data)
    nEnsemble = length(θ)

    θ_series = [copy(θ)]

    error = norm(model_data - ideal_data)

    @info("\nIteration 0 with error $error")

	G = [copy(model_data) for i in 1:nEnsemble]

	# EKI iteration is equivalent to a time step of the above equation.
    @inbounds for i in 1:nIterations
        θ̄ = mean(θ)

		# Evaluating the forward map for all ensemble members. This is the most expensive step because it needs to run
        # the model nEnsemble times. For the moment our model is simple, but imagine doing this with a full climate
        # model! Luckily this step is embarassingly parallelizeable.
        Threads.@threads for n in 1:nEnsemble
			G[n] .= forward_map(Nx, Ny, spacing_type, θ[n]) # Error handling needs to go here.
		end

		# The ensemble mean output of the models
		G̅ = mean(G)

        # Calculating the covariances to be used in the update steps
        Cᵘᵖ = (θ[1] - θ̄) * (G[1] - G̅)'
        Cᵖᵖ = (G[1] - G̅) * (G[1] - G̅)'

        for j = 2:nEnsemble
            Cᵘᵖ += (θ[j] - θ̄) * (G[j] - G̅)'
            Cᵖᵖ += (G[j] - G̅) * (G[j] - G̅)'
        end

        Cᵘᵖ *= 1 / (nEnsemble - 1)
        Cᵖᵖ *= 1 / (nEnsemble - 1)

        # Ensemblize the data (adding the random noise η).
        y = [ideal_data + Δt * randn(nData) for i in 1:nEnsemble]

		# The residual from our observations
        r = y - G

        # Update the parameters using implicit pseudo-time-stepping, which involves solving a linear system.
        Cᵖᵖ_factorized = cholesky(Symmetric(Cᵖᵖ + 1 / Δt * LinearAlgebra.I))

        for j in 1:nEnsemble
            θ[j] .+= Cᵘᵖ * (Cᵖᵖ_factorized \ r[j])
        end

        error = norm(mean(r))
        @info "Iteration $i with error $error"
        push!(θ_series, copy(θ))
    end

    return θ_series
end

"""
    optimized_non_uniform_conformal_cubed_sphere_coordinates(Nx, Ny, spacing_type)

High-level driver that uses EKI to optimize the non-uniform spacing parameter for a conformal cubed sphere panel, then 
builds and returns the corresponding grid.

Procedure:
1. Create a random ensemble of parameters within limits (`nEnsemble = 40`, reproducible seed).
2. Run `optimize!` to fit the **weighted** diagnostics to their ideal targets.
3. Build the optimized grid with `conformal_cubed_sphere_coordinates(Nx, Ny; non_uniform_spacing=true, ...)` using the 
   mean optimized parameter.

For `"geometric"`, the parameter is `ratio^(Nx-1)`; for `"exponential"`, the parameter is `k₀/Nx`.

# Arguments
- `Nx, Ny`: Number of grid vertices along panel coordinates.
- `spacing_type`: `"geometric"` or `"exponential"`.

# Returns
- `x, y`: Computational-space coordinates of lengths `Nx` and `Ny`.
- `X, Y, Z`: `(Nx, Ny)` Cartesian coordinates of the vertices of the **optimized** non-uniform conformal cubed sphere
  panel.
"""
function optimized_non_uniform_conformal_cubed_sphere_coordinates(Nx, Ny, spacing_type)
    nEnsemble = 40 # Choose nEnsemble to be at least 4 times the number of parameters.
    
    @info "Optimize non-uniform conformal cubed sphere for Nx = $Nx and Ny = $Ny"
    
    begin
        Random.seed!(123)
        θᵣ = specify_random_parameters(nEnsemble, spacing_type)
        θᵢ = deepcopy(θᵣ)

        θ_series = optimize!(Nx, Ny, spacing_type, θᵣ; nIterations = 10)
    end
    
    if spacing_type == "geometric"
        θ_name = "ratio_raised_to_Nx_minus_one"
    elseif spacing_type == "exponential"
        θ_name = "k₀ByNx"
    end

    println("\nThe unoptimized parameters are: $θ_name = $(round(mean(θᵢ)[1], digits=2))\n")
    println("\nThe optimized parameters are: $θ_name = $(round(mean(θᵣ)[1], digits=2))\n")
    
    x, y, X, Y, Z = (
    conformal_cubed_sphere_coordinates(Nx, Ny; non_uniform_spacing = true, spacing_type = spacing_type,
                                       ratio_raised_to_Nx_minus_one = mean(θᵣ)[1], k₀ByNx = mean(θᵣ)[1]))
    
    return x, y, X, Y, Z
end
