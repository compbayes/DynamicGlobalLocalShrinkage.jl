using Pkg

# Activate the docs environment in docs/Project.toml
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()  # optional but good to keep

# With minor changes from https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/docs

### Process examples
# Always rerun examples
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)

# Install and precompile all packages
# Workaround for https://github.com/JuliaLang/Pkg.jl/issues/2219
examples = filter!(isdir, readdir(joinpath(@__DIR__, "..", "examples"); join=true))
above = joinpath(@__DIR__, "..")
let script = "using Pkg; Pkg.activate(ARGS[1]); Pkg.develop(PackageSpec(path=pwd()))
;Pkg.instantiate();"
    for example in examples
        if !success(`$(Base.julia_cmd()) -e $script $example`)
            error("project environment of example ", basename(example), " could not be instantiated",)
        end
    end
end
# Run examples asynchronously
processes = let literatejl = joinpath(@__DIR__, "literate.jl")
    map(examples) do example
        return run(
            pipeline(
                `$(Base.julia_cmd()) $literatejl $(basename(example)) $EXAMPLES_OUT`;
                stdin=devnull,
                stdout=devnull,
                stderr=stderr,
            );  
            wait=false,
        )::Base.Process
    end
end

# Check that all examples were run successfully
isempty(processes) || success(processes) || error("some examples were not run successfully")
println("All examples were run successfully")

using Pkg
Pkg.activate("./docs/")
using Documenter
using DynamicGlobalLocalShrinkage

DocMeta.setdocmeta!(DynamicGlobalLocalShrinkage, :DocTestSetup, 
    :(using DynamicGlobalLocalShrinkage); recursive=true)

makedocs(;
    sitename="DynamicGlobalLocalShrinkage.jl",
    modules=[DynamicGlobalLocalShrinkage],
    authors="Mattias Villani",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://compbayes.github.io/DynamicGlobalLocalShrinkage.jl",
        edit_link="main",
        assets=String[],
        size_threshold_warn = 1000 * 2^10,
        size_threshold = 1500 * 2^10, # 1000 KiB determines the maximal html size in KiB
    ),
    

    pages = [
        "Home" => "index.md",
        "Sampling Updates" => "samplingUpdates.md",
        "Index" => "functionindex.md",
        "Examples" => [
            map(
                (x) -> joinpath("examples", x),
                filter!(filename -> endswith(filename, ".md"), readdir(EXAMPLES_OUT)),
            )...,
        ],
    ],
)

deploydocs(;
    repo="github.com/compbayes/DynamicGlobalLocalShrinkage.jl.git",
)

