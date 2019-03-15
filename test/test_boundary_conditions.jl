function test_diffusion_simple(fld)
    Nx, Ny, Nz = 1, 1, 16
    Lx, Ly, Lz = 1, 1, 1
    κ = 1
    eos = LinearEquationOfState(βS=0, βT=0)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=κ, κ=κ, eos=eos)

    if fld ∈ (:u, :v, :w)
        field = getfield(model.velocities, fld)
    else
        field = getfield(model.tracers, fld)
    end

    value = π
    field.data .= value

    Δt = 0.01 # time-step much less than diffusion time-scale
    Nt = 10
    time_step!(model, Nt, Δt)

    !any(@. !isapprox(value, field.data))
end


function test_diffusion_budget(field_name)
    Nx, Ny, Nz = 1, 1, 16
    Lx, Ly, Lz = 1, 1, 1
    κ = 1
    eos = LinearEquationOfState(βS=0, βT=0)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=κ, κ=κ, eos=eos)

    if field_name ∈ (:u, :v, :w)
        field = getfield(model.velocities, field_name)
    else
        field = getfield(model.tracers, field_name)
    end

    half_Nz = floor(Int, Nz/2)
    @views @. field.data[:, :, 1:half_Nz]  = -1
    @views @. field.data[:, :, half_Nz:end] =  1

    mean_init = mean(field.data)
    τκ = Lz^2 / κ # diffusion time-scale
    Δt = 0.0001 * τκ # time-step much less than diffusion time-scale
    Nt = 100

    time_step!(model, Nt, Δt)
    isapprox(mean_init, mean(field.data))
end

function test_diffusion_cosine(fld)
    Nx, Ny, Nz = 1, 1, 16
    Lx, Ly, Lz = 1, 1, π/2
    κ = 1
    eos = LinearEquationOfState(βS=0, βT=0)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=κ, κ=κ, eos=eos)
    @show model.grid.zF[1] model.grid.zF[end]

    if fld == :w
        throw("There are no boundary condition tests for w yet.")
    elseif fld ∈ (:u, :v)
        field = getfield(model.velocities, fld)
    else
        field = getfield(model.tracers, fld)
    end

    zC = model.grid.zC
    m = 2
    @views @. field.data[1, 1, :] = cos.(m*z)
    field_ans(z, t) = exp(-κ*m^2*t) * cos(m*z)

    τκ = Lz^2 / κ # diffusion time-scale
    Δt = 1e-6 * τκ # time-step much less than diffusion time-scale
    Nt = 100

    time_step!(model, Nt, Δt)

    field_num = dropdims(field.data, dims=(1, 2))

    @show norm(field_num .- field_ans.(zC, model.clock.time))

    false
end
