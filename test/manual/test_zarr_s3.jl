#####
##### Manual smoke test: ZarrWriter against an S3-compatible store (e.g., MinIO).
#####
##### Not run in CI. To exercise:
#####
#####   1. Start a local MinIO:
#####        docker run -p 9000:9000 -p 9001:9001 \
#####            -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
#####            quay.io/minio/minio server /data --console-address ":9001"
#####
#####   2. Create a bucket via the console at http://localhost:9001 (`oceananigans-test`).
#####
#####   3. Run:
#####        OCEANANIGANS_TEST_S3=1 \
#####            AWS_ACCESS_KEY_ID=minioadmin \
#####            AWS_SECRET_ACCESS_KEY=minioadmin \
#####            julia --project=test test/manual/test_zarr_s3.jl
#####

if get(ENV, "OCEANANIGANS_TEST_S3", "0") != "1"
    @info "Skipping S3 smoke test. Set OCEANANIGANS_TEST_S3=1 to run."
    exit(0)
end

using Test
using Oceananigans
using Zarr
using AWS

# Adjust these for your environment.
const ENDPOINT = get(ENV, "ZARR_S3_ENDPOINT", "http://localhost:9000")
const BUCKET   = get(ENV, "ZARR_S3_BUCKET",   "oceananigans-test")
const PREFIX   = "zarr_writer_smoke"

# Build the S3 store.
store = Zarr.S3Store(BUCKET, PREFIX; aws_endpoint=ENDPOINT)

grid = RectilinearGrid(CPU(), size=(8, 8, 8), extent=(1, 1, 1),
                       topology=(Periodic, Periodic, Periodic))
model = NonhydrostaticModel(grid; tracers=:c)
set!(model, u=(x, y, z) -> 1.0)

simulation = Simulation(model, Δt=1.0, stop_iteration=3)
simulation.output_writers[:s3] = ZarrWriter(model, (; u=model.velocities.u);
                                            store = store,
                                            schedule = IterationInterval(1),
                                            overwrite_existing = true)
run!(simulation)

g = Zarr.zopen(store)
@test length(g["time"][:]) == 4
@test "u" in keys(g.arrays)
@info "S3 smoke test PASSED"
