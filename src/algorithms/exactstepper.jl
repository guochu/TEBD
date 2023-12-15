
struct ExactStepper{T <: Number} <: AbstractStepper
	tspan::Tuple{T, T}
	ishermitian::Bool
end

function Base.getproperty(x::ExactStepper, s::Symbol)
	if s == :δ
		return x.tspan[2] - x.tspan[1]
	else
		getfield(x, s)
	end
end

"""
	ExactStepper(; tspan::Tuple{<:Number, <:Number}) 
"""
function ExactStepper(; tspan::Tuple{<:Number, <:Number}, ishermitian::Bool) 
	ti, tf = tspan
	T = promote_type(typeof(ti), typeof(tf))
	return ExactStepper((convert(T, ti), convert(T, tf)), ishermitian)
end

Base.similar(x::ExactStepper; ishermitian::Bool=x.ishermitian, tspan::Tuple{<:Number, <:Number}) = ExactStepper(tspan=tspan, ishermitian=ishermitian)

mutable struct HomogeousExactCache{E<:ExactCache, S<:ExactStepper} <: AbstractTimeEvoCache
	env::E
	stepper::S
end

HomogeousExactCache(h::Union{MPO, MPOHamiltonian}, stepper::ExactStepper, state::ExactMPS) = HomogeousExactCache(environments(h, state), stepper)

function recalculate!(x::HomogeousExactCache, mpo::AbstractMPO, stepper::ExactStepper, state::ExactMPS)
	if !((x.env.state === state) && (x.env.mpo === mpo))
		return HomogeousExactCache(environments(mpo, state), stepper)
	else
		return HomogeousExactCache(x.env, stepper)
	end
end

function make_step!(mpo::AbstractMPO, stepper::ExactStepper, state::ExactMPS, x::HomogeousExactCache) 
	x = recalculate!(x, mpo, stepper, state)
	env = exact_timeevolution!(x.env, stepper.δ, ishermitian=stepper.ishermitian)
	return env.state, x
end

mutable struct HomogeousHamExactCache{H<:QuantumOperator, E<:ExactCache, S<:ExactStepper} <: AbstractTimeEvoCache
	h::H
	env::E
	stepper::S
end

function HomogeousExactCache(h::QuantumOperator, stepper::ExactStepper, state::ExactMPS) 
	mpo = MPO(h)
	return HomogeousHamExactCache(h, environments(mpo, state), stepper)
end

function recalculate!(x::HomogeousHamExactCache, h::QuantumOperator, stepper::ExactStepper, state::ExactMPS) 
	if !((x.env.state === state) && (x.h === h))
		return HomogeousExactCache(h, stepper, state)
	else
		return HomogeousHamExactCache(x.h, x.env, stepper)
	end	
end

function make_step!(h::QuantumOperator, stepper::ExactStepper, state::ExactMPS, x::HomogeousHamExactCache) 
	x = recalculate!(x, h, stepper, state)
	env = exact_timeevolution!(x.env, stepper.δ, ishermitian=stepper.ishermitian)
	return env.state, x
end


# exact
timeevo_cache(h::Union{MPO, MPOHamiltonian, QuantumOperator}, stepper::ExactStepper, state::ExactMPS) = HomogeousExactCache(h, stepper, state)
timeevo_cache(h::Union{MPO, MPOHamiltonian, QuantumOperator}, stepper::ExactStepper, state::MPS) = timeevo_cache(h, stepper, ExactMPS(state))

# exact
timeevo!(state::ExactMPS,  h::Union{MPO, MPOHamiltonian, QuantumOperator}, stepper::ExactStepper, 
	cache=timeevo_cache(h, stepper, state)) = make_step!(h, stepper, state, cache)

function timeevo!(state::MPS,  h::Union{MPO, MPOHamiltonian, QuantumOperator}, stepper::ExactStepper)
	cache=timeevo_cache(h, stepper, state)
	return make_step!(h, stepper, cache.state, cache)
end 
