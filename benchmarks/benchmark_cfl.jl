using BenchmarkTools

using Oceananigans
using Oceananigans.Architectures
using JULES

Archs = [CPU]
Ns = [32]
Tvars = [Energy, Entropy]
Gases = [DryEarth, DryEarth3]

tags = ["arch", "N", "gases", "tvar"]

suite = BenchmarkGroup(
    "cfl" => BenchmarkGroup(),
    "acoustic_cfl" => BenchmarkGroup()
)

for Arch in Archs, N in Ns, Gases in Gases, Tvar in Tvars
    @info "Running CFL benchmark [$Arch, N=$N, $Tvar, $Gases]..."

    grid = RegularCartesianGrid(size=(N, N, N), extent=(1, 1, 1))
    model = CompressibleModel(architecture=Arch(), grid=grid, thermodynamic_variable=Tvar(),
                              gases=Gases())

    # warmup
    cfl(model, 1)
    acoustic_cfl(model, 1)

    b_cfl = @benchmark cfl($model, 1) samples=10
    display(b_cfl)

    b_acfl = @benchmark acoustic_cfl($model, 1) samples=10
    display(b_acfl)

    key = (Arch, N, Gases, Tvar)
    suite["cfl"][key] = b_cfl
    suite["acoustic_cfl"][key] = b_acfl
end


