include("ars.jl")
include("ars_defs.jl")
include("probability_utils.jl")


function gaussian(x; sigma=1)
    log(exp(-x ^ 2 / sigma))
end

function halfgaussian(x; sigma=3)
    log(exp(-x ^ 2 / sigma)) * (1 * (x <= 0) + 1e300 * (x > 0))
end

function rel_p_logpdf(p; m=1., c=1.)
    -m*c^2*sqrt(p^2/(m^2*c^2)+1)
end

# println(ARGS)
name = ARGS[1]
a = parse(Float64, ARGS[2])
b = parse(Float64, ARGS[3])
nSamples = parse(Int64, ARGS[4])

domain_left = ARGS[5]
domain_right = ARGS[6]

debug_output = ARGS[7]

if domain_left == "inf"
    domain_left = Inf
elseif domain_left == "-inf"
    domain_left = -Inf
else
    domain_left = parse(Float64, domain_left)
end

if domain_right == "inf"
    domain_right = Inf
elseif domain_right == "-inf"
    domain_right = -Inf
else
    domain_right = parse(Float64, domain_right)
end


# Change this to parse from args
domain = [domain_left, domain_right]

n_rand_calls = 0
function rand()
    cmd = `python -c "from numpy.random import rand, seed; seed(1); [rand() for _ in range($n_rand_calls)]; print(rand())"`
    global n_rand_calls += 1
    rand_val = readall(cmd)
    return parse(Float64, rand_val)
end

if name == "1d-gaussian"
   result = ars(gaussian, a, b, domain, nSamples, debug_output) 
elseif name == "1d-half-gaussian"
   result = ars(halfgaussian, a, b, domain, nSamples, debug_output) 
elseif name == "relativistic_momentum_logpdf"
    result = ars(rel_p_logpdf, a, b, domain, nSamples, debug_output)
end
println(result)

