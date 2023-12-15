abstract type AbstractStepper end

abstract type AbstractTimeEvoCache <: AbstractCache end


# TEBD
struct TEBDStepper{T<:Number, C <: TruncationScheme} <: AbstractStepper
	tspan::Tuple{T, T}
	stepsize::T
	n::Int
	n₀::Int
	order::Int
	trunc::C
end

"""
	TEBDStepper(;stepsize, tspan=(0., stepsize), order=2, trunc=DefaultTruncation)

Return a TEBDStepper using Trotter expansion of 'order'.

n can be divided by n₀, δ = tspan[2] - tspan[1] = n * stepsize,
n₀ steps of the quantum circuit is stored to limit memory usage.
"""
function TEBDStepper(;stepsize::Number, tspan::Tuple{<:Number, <:Number}=(0., stepsize), order::Int=2, n₀::Int=4, trunc::TruncationScheme=DefaultTruncation)
	ti, tf = tspan
	δ = tf - ti
	n, stepsize, n₀ = compute_step_size(δ, stepsize, n₀)
	T = promote_type(typeof(ti), typeof(tf), typeof(stepsize))
	return TEBDStepper((convert(T, ti), convert(T, tf)), convert(T, stepsize), n, n₀, order, trunc)
end

function Base.getproperty(x::TEBDStepper, s::Symbol)
	if s == :δ
		return x.tspan[2] - x.tspan[1]
	else
		getfield(x, s)
	end
end

"""
	Base.similar(x::AbstractStepper; tspan::Tuple{<:Number, <:Number}, stepsize::Number=x.stepsize)

Retuen a similar stepper from existing stepper, possible changing tspan and stepsize
"""
Base.similar(x::TEBDStepper; tspan::Tuple{<:Number, <:Number}, stepsize::Number=x.stepsize) = TEBDStepper(
	tspan=tspan, stepsize=stepsize, order=x.order, n₀=x.n₀, trunc=x.trunc)


struct TDVPStepper{T<:Number, A<:TDVPAlgorithm} <: AbstractStepper
	tspan::Tuple{T, T}
	n::Int
	alg::A	
end

function Base.getproperty(x::TDVPStepper, s::Symbol)
	if s == :δ
		return x.tspan[2] - x.tspan[1]
	elseif s == :order
		return 2
	elseif s == :stepsize
		return x.alg.stepsize
	else
		getfield(x, s)
	end
end

function _change_stepsize(x::TDVP1, stepsize::Number)
	if !(x.stepsize ≈ stepsize)
		x = TDVP1(stepsize=stepsize, D=x.D, exptol=x.exptol, ishermitian=x.ishermitian, verbosity=x.verbosity)
	end
	return x
end
function _change_stepsize(x::TDVP2, stepsize::Number)
	if !(x.stepsize ≈ stepsize)
		x = TDVP2(stepsize=stepsize, exptol=x.exptol, ishermitian=x.ishermitian, trunc=x.trunc, verbosity=x.verbosity)
	end
	return x
end
function _change_stepsize(x::TDVP1S, stepsize::Number)
	if !(x.stepsize ≈ stepsize)
		x = TDVP1S(stepsize=stepsize, exptol=x.exptol, ishermitian=x.ishermitian, trunc=x.trunc, expan=x.expan, verbosity=x.verbosity)
	end
	return x
end

"""
	TDVPStepper(;alg::TDVPAlgorithm, tspan)

Return a TDVPStepper
"""
function TDVPStepper(;alg::TDVPAlgorithm, tspan::Tuple{<:Number, <:Number}=(0., alg.stepsize))
	ti, tf = tspan
	δ = tf - ti	
	n, stepsize = compute_step_size(δ, alg.stepsize)
	T = promote_type(typeof(ti), typeof(tf), typeof(stepsize))
	return TDVPStepper((convert(T, ti), convert(T, tf)), n, _change_stepsize(alg, stepsize))
end

Base.similar(x::TDVPStepper; tspan::Tuple{<:Number, <:Number}, stepsize::Number=x.stepsize) = TDVPStepper(tspan=tspan, alg=_change_stepsize(x.alg, stepsize))


mutable struct HomogeousTEBDCache{H<:QuantumOperator, C<:QuantumCircuit, S<:TEBDStepper} <: AbstractTimeEvoCache
	h::H
	circuit::C
	stepper::S
end

function HomogeousTEBDCache(h::QuantumOperator, stepper::TEBDStepper)
	isconstant(h) || throw(ArgumentError("const hamiltonian expected."))
	circuit = fuse_gates(repeat(trotter_propagator(h, (0., stepper.stepsize), order=stepper.order, stepsize=stepper.stepsize), stepper.n₀))
	return HomogeousTEBDCache(h, circuit, stepper)
end

"""
	recalculate!(x::AbstractTimeEvoCache, h, stepper)

Recalculate the cache with possible new Hamiltonian and new stepper
"""
function recalculate!(x::HomogeousTEBDCache{H}, h::H, stepper::TEBDStepper) where H
	if !((x.h === h) && (stepper.δ==x.stepper.δ) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) && (stepper.n₀==x.stepper.n₀) )
		isconstant(h) || throw(ArgumentError("const hamiltonian expected."))
		return HomogeousTEBDCache(h, fuse_gates(repeat(trotter_propagator(h, (0., stepper.stepsize), order=stepper.order, stepsize=stepper.stepsize), stepper.n₀)), stepper)
	else
		return x
	end
end

"""
	make_step!(h, stepper, state, cache)

Evolving the quantum state from tspan[1] to tspan[2] using the stepper, the
cache will also be updated.
"""
function make_step!(h::H, stepper::TEBDStepper, state::MPS, x::HomogeousTEBDCache{H}) where H
	x = recalculate!(x, h, stepper)
	@assert (stepper.n % stepper.n₀ == 0)
	m = div(stepper.n, stepper.n₀)
	for i in 1:m
		apply!(x.circuit, state, trunc=x.stepper.trunc)
	end
	return state, x
end


mutable struct InhomogenousTEBDCache{H<:QuantumOperator, S<:TEBDStepper} <: AbstractTimeEvoCache
	h::H
	# circuit::C
	stepper::S	
end


function recalculate!(x::InhomogenousTEBDCache{H}, h::H, stepper::TEBDStepper) where {H}
	if !((x.h === h) && (stepper.tspan==x.stepper.tspan) && (stepper.stepsize==x.stepper.stepsize) && (stepper.order==x.stepper.order) && (stepper.n₀==x.stepper.n₀) )
		return InhomogenousTEBDCache(h, stepper)
	else
		return x
	end
end

function make_step!(h::H, stepper::TEBDStepper, state::MPS, x::InhomogenousTEBDCache{H}) where H
	x = recalculate!(x, h, stepper)
	ti, tf = x.stepper.tspan
	stepsize = x.stepper.stepsize
	@assert (x.stepper.n % x.stepper.n₀ == 0)
	m = div(x.stepper.n, x.stepper.n₀)
	# δt = x.stepper.δ / m
	δt = stepsize * x.stepper.n₀
	for i in 1:m
		tf = ti + δt
		circuit = fuse_gates(trotter_propagator(x.h, (ti, tf), order=stepper.order, stepsize=stepper.stepsize))
		apply!(circuit, state, trunc=x.stepper.trunc)
		ti = tf
	end
	return state, x
end

TEBDCache(h::QuantumOperator, stepper::TEBDStepper) = isconstant(h) ? HomogeousTEBDCache(h, stepper) : InhomogenousTEBDCache(h, stepper)

# TDVP
mutable struct HomogeousTDVPCache{E<:ExpectationCache, S<:TDVPStepper} <: AbstractTimeEvoCache
	env::E
	stepper::S
end

HomogeousTDVPCache(h::Union{MPO, MPOHamiltonian}, stepper::TDVPStepper, state::MPS) = HomogeousTDVPCache(environments(h, state), stepper)
TDVPCache(h::Union{MPO, MPOHamiltonian}, stepper::TDVPStepper, state::MPS) = HomogeousTDVPCache(h, stepper, state)


function recalculate!(x::HomogeousTDVPCache, h::Union{MPO, MPOHamiltonian}, stepper::TDVPStepper, state::MPS)
	if !((x.env.state === state) && (x.env.h === h))
		return HomogeousTDVPCache(environments(h, state), stepper)
	else
		return HomogeousTDVPCache(x.env, stepper)
	end
end

function make_step!(h::Union{MPO, MPOHamiltonian}, stepper::TDVPStepper, state::MPS, x::HomogeousTDVPCache)
	x = recalculate!(x, h, stepper, state)
	for i in 1:stepper.n
		sweep!(x.env, stepper.alg)
	end
	return state, x
end

mutable struct HomogeousHamTDVPCache{H<:QuantumOperator, E<:ExpectationCache, S<:TDVPStepper} <: AbstractTimeEvoCache
	h::H
	env::E
	stepper::S
end
function HomogeousTDVPCache(h::QuantumOperator, stepper::TDVPStepper, state::MPS) 
	mpo = MPO(h)
	env = environments(mpo, state)
	return HomogeousHamTDVPCache(h, env, stepper)
end

function recalculate!(x::HomogeousHamTDVPCache{H}, h::H, stepper::TDVPStepper, state::MPS) where H
	if !((x.env.state === state) && (x.h === h))
		return HomogeousTDVPCache(h, stepper, state)
	else
		return HomogeousHamTDVPCache(x.h, x.env, stepper)
	end	
end

function make_step!(h::H, stepper::TDVPStepper, state::MPS, x::HomogeousHamTDVPCache{H}) where H
	x = recalculate!(x, h, stepper, state)
	for i in 1:stepper.n
		sweep!(x.env, stepper.alg)
	end
	return state, x
end

mutable struct InhomogenousHamTDVPCache{H<:QuantumOperator, S<:TDVPStepper} <: AbstractTimeEvoCache
	h::H
	stepper::S
end

InhomogenousTDVPCache(h::QuantumOperator, stepper::TDVPStepper, state::MPS) = InhomogenousHamTDVPCache(h, stepper)

function recalculate!(x::InhomogenousHamTDVPCache{H}, h::H, stepper::TDVPStepper, state::MPS) where H
	return InhomogenousHamTDVPCache(h, stepper)
end

function make_step!(h::H, stepper::TDVPStepper, state::MPS, x::InhomogenousHamTDVPCache{H}) where H
	x = recalculate!(x, h, stepper, state)
	t_start = stepper.tspan[1]
	for i in 1:stepper.n
		t = t_start + (i-1) * stepper.stepsize + stepper.stepsize/2
		mpo = MPO(x.h(t))
		env = environments(mpo, state)
		sweep!(env, stepper.alg)
	end
	return state, x
end

function TDVPCache(h::QuantumOperator, stepper::TDVPStepper, state::MPS) 
	isconstant(h) ? HomogeousTDVPCache(h, stepper, state) : InhomogenousTDVPCache(h, stepper, state)
end


"""
	timeevo_cache(h, stepper, state)

Return a cache for time evolution.

The following dispatches are supported:
* timeevo_cache(h::QuantumOperator, stepper::TEBDStepper, state::MPS);
* timeevo_cache(h::Union{MPO, MPOHamiltonian}, stepper::TDVPStepper, state::MPS);
* timeevo_cache(h::QuantumOperator, stepper::TDVPStepper, state::MPS);


This function can be explicitly used, or implicitly used with function timeevo!
"""
timeevo_cache(h::Union{QuantumOperator, MPO, MPOHamiltonian}, stepper::AbstractStepper, state::AbstractMPS) = error(
	"timeevo_cache not implemented for types $((typeof(h), typeof(stepper), typeof(state)))")
timeevo_cache(h::QuantumOperator, stepper::TEBDStepper, state::MPS) = TEBDCache(h, stepper)
timeevo_cache(h::Union{MPO, MPOHamiltonian}, stepper::TDVPStepper, state::MPS) = TDVPCache(h, stepper, state)
timeevo_cache(h::QuantumOperator, stepper::TDVPStepper, state::MPS) = TDVPCache(h, stepper, state)



"""
	timeevo!(state, h, stepper, cache=timeevo_cache(h, stepper, state)) -> state, cache

Inplace evolving the state using the Hamiltonian h and the stepper
from time tspan[1] to time tspan[2], tspan is a field of stepper

Cache will be created if not input explicitly, the state and the 
cache will both be returned, where the cache could be used to speedup 
the next time evolution.

The following dispatches are supported:
* timeevo!(state::MPS, h::QuantumOperator, stepper::TEBDStepper, cache=timeevo_cache(h, stepper, state));
* timeevo!(state::MPS, h::Union{MPO, MPOHamiltonian, QuantumOperator}, stepper::TDVPStepper, cache=timeevo_cache(h, stepper, state));
"""
timeevo!(state::MPS, h::QuantumOperator, stepper::TEBDStepper, cache=timeevo_cache(h, stepper, state)) = make_step!(h, stepper, state, cache)
timeevo!(state::MPS, h::Union{MPO, MPOHamiltonian, QuantumOperator}, stepper::TDVPStepper, cache=timeevo_cache(h, stepper, state)) = make_step!(
	h, stepper, state, cache)


function compute_step_size(t::Number, dt::Number)
	n = ceil(Int, abs(t / dt)) 
	@assert (n > 0)
	if n == 0
		n = 1
	end
	return n, t / n
end

function compute_step_size(t::Number, dt::Number, n₀::Int)
	n = ceil(Int, abs(t / dt)) 
	n₀ = min(n, n₀)
	m = ceil(Int, n / n₀) 
	n = m * n₀
	@assert (n > 0)
	return n, t / n, n₀
end
