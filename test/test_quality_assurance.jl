using Oceananigans: Oceananigans
using Aqua: Aqua
using ExplicitImports: ExplicitImports
using Test: @testset, @test, detect_ambiguities

@testset "Aqua" begin
    @info "testing quality assurance via Aqua"
    Aqua.test_all(Oceananigans; ambiguities=false)

    # Until we resolve all ambiguities, we make sure we don't increase them.
    # Do not increase this number. If ambiguities increase, resolve them before merging.
    number_of_ambiguities = length(detect_ambiguities(Oceananigans; recursive=true))
    @test number_of_ambiguities <= 338
    @info "Number of ambiguities: $number_of_ambiguities"

    modules = (
        # Oceananigans.AbstractOperations,
        # Oceananigans.Advection,
        Oceananigans.Architectures,
        Oceananigans.Biogeochemistry,
        # Oceananigans.BoundaryConditions,
        Oceananigans.BuoyancyFormulations,
        Oceananigans.Coriolis,
        Oceananigans.Diagnostics,
        # Oceananigans.DistributedComputations,
        # Oceananigans.Fields,
        Oceananigans.Forcings,
        # Oceananigans.Grids,
        # Oceananigans.ImmersedBoundaries,
        Oceananigans.Logger,
        # Oceananigans.Models,
        # Oceananigans.MultiRegion,
        # Oceananigans.Operators,
        # Oceananigans.OrthogonalSphericalShellGrids,
        # Oceananigans.OutputReaders,
        # Oceananigans.OutputWriters,
        Oceananigans.Simulations,
        # Oceananigans.Solvers,
        Oceananigans.StokesDrifts,
        Oceananigans.TimeSteppers,
        # Oceananigans.TurbulenceClosures,
        Oceananigans.Units,
        # Oceananigans.Utils,
    )

    # In addition to capping the total number of ambiguities above, we make sure
    # modules which don't have any don't get new ones.
    @testset "No ambiguities for module $(mod)" for mod in modules
        @info "Testing no ambiguities for module $(mod)"
        @test isempty(detect_ambiguities(mod; recursive=true))
    end
end

@testset "ExplicitImports" begin

    modules = (
        Oceananigans.AbstractOperations,
        # Oceananigans.Advection,
        # Oceananigans.Architectures,
        Oceananigans.Biogeochemistry,
        # Oceananigans.BoundaryConditions,
        Oceananigans.BuoyancyFormulations,
        Oceananigans.Coriolis,
        Oceananigans.Diagnostics,
        # Oceananigans.DistributedComputations,
        Oceananigans.Fields,
        Oceananigans.Forcings,
        # Oceananigans.Grids,
        Oceananigans.ImmersedBoundaries,
        # Oceananigans.Logger,
        # Oceananigans.Models,
        Oceananigans.Models.HydrostaticFreeSurfaceModels,
        Oceananigans.MultiRegion,
        Oceananigans.Operators,
        Oceananigans.OrthogonalSphericalShellGrids,
        # Oceananigans.OutputReaders,
        # Oceananigans.OutputWriters,
        # Oceananigans.Simulations,
        Oceananigans.Solvers,
        # Oceananigans.StokesDrifts,
        Oceananigans.TimeSteppers,
        Oceananigans.TurbulenceClosures,
        Oceananigans.Units,
        Oceananigans.Utils,
    )

    @testset "Explicit Imports [$(mod)]" for mod in modules
        @info "Testing no implicit imports for module $(mod)"
        @test ExplicitImports.check_no_implicit_imports(mod) === nothing
    end

    @testset "Import via Owner" begin
        @info "Testing imports via owner"
        @test ExplicitImports.check_all_explicit_imports_via_owners(Oceananigans) === nothing
    end

    @testset "Stale Explicit Imports" begin
        @info "Testing no stale implicit imports"
        @test ExplicitImports.check_no_stale_explicit_imports(Oceananigans) === nothing
    end

    @testset "Qualified Accesses" begin
        @info "Testing qualified access via owners"
        @test ExplicitImports.check_all_qualified_accesses_via_owners(Oceananigans) === nothing
    end

    @testset "Self Qualified Accesses" begin
        @info "Testing no self qualified accesses"
        @test ExplicitImports.check_no_self_qualified_accesses(Oceananigans) === nothing
    end
end

################################################################################

# Code for detecting `Core.Box`es adapted from
# <https://github.com/JuliaLang/julia/pull/60478>.  There's a chance this will be eventually
# integrated in Aqua, for the time being we vendor the code here.

function is_box_call(expr)
    if !(expr isa Expr)
        return false
    end
    if expr.head === :call
        callee = expr.args[1]
        return callee === Core.Box || (callee isa GlobalRef && callee.mod === Core && callee.name === :Box)
    elseif expr.head === :new
        callee = expr.args[1]
        return callee === Core.Box || (callee isa GlobalRef && callee.mod === Core && callee.name === :Box)
    end
    return false
end

function slot_name(ci, slot)
    if slot isa Core.SlotNumber
        idx = Int(slot.id)
        if 1 <= idx <= length(ci.slotnames)
            return string(ci.slotnames[idx])
        end
    end
    return string(slot)
end

function method_location(m::Method)
    file = m.file
    line = m.line
    file_str = file isa Symbol ? String(file) : string(file)
    if file_str == "none" || line == 0
        return ("", 0)
    end
    return (file_str, line)
end

function root_module(mod::Module)
    while true
        parent = parentmodule(mod)
        if parent === mod || parent === Main || parent === Core
            return mod
        end
        mod = parent
    end
end

function format_box_fields(var, m::Method)
    file, line = method_location(m)
    location = isempty(file) ? "" : string(file, ":", line)
    return (
        mod = string(root_module(m.module)),
        var = string(var),
        func = string(m.name),
        sig = string(m.sig),
        location = location,
    )
end

function escape_md(s)
    return replace(string(s), "|" => "\\|")
end

function md_code(s)
    return "`" * replace(string(s), "`" => "``") * "`"
end

function scan_method!(lines, m::Method, modules)
    root = string(root_module(m.module))
    if !isempty(modules) && !(root in modules)
        return
    end
    ci = try
        Base.uncompressed_ast(m)
    catch
        return
    end
    for stmt in ci.code
        if stmt isa Expr && stmt.head === :(=)
            lhs = stmt.args[1]
            rhs = stmt.args[2]
            if is_box_call(rhs)
                push!(lines, format_box_fields(slot_name(ci, lhs), m))
            end
        elseif is_box_call(stmt)
            push!(lines, format_box_fields("<unknown>", m))
        end
    end
end

function check_no_boxes()
    modules = Set(["Oceananigans"])
    format = "markdown"
    lines = Vector{NamedTuple}()
    Base.visit(Core.methodtable) do m
        scan_method!(lines, m, modules)
    end
    sort!(lines, by = entry -> (entry.mod, entry.func, entry.var))
    if format == "plain"
        for entry in lines
            println("mod=", entry.mod,
                    "\tvar=", entry.var,
                    "\tfunc=", entry.func,
                    "\tsig=", entry.sig,
                    "\tlocation=", entry.location)
        end
    else
        # treat "markdown" and "markdown-table" as table output
        last_mod = ""
        for entry in lines
            if entry.mod != last_mod
                if !isempty(last_mod)
                    println()
                end
                println("## $(length(lines)) `Core.Box`es detected in module `", entry.mod, "`")
                println("| var | func | sig | location |")
                println("| --- | --- | --- | --- |")
                last_mod = entry.mod
            end
            println("| ", md_code(escape_md(entry.var)),
                    " | ", md_code(escape_md(entry.func)),
                    " | ", md_code(escape_md(entry.sig)),
                    " | ", md_code(escape_md(entry.location)),
                    " |")
        end
    end

    return isempty(lines)
end

################################################################################

@testset "No Core.Box" begin
    # Too complicated to adapt to v1.11-, skip it in that case.
    @test check_no_boxes() skip=(VERSION < v"1.12")
end
