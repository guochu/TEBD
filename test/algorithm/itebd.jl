println("------------------------------------")
println("----------|    iTEBD    |-----------")
println("------------------------------------")


function spin_site_ops_u1x()
    ph = Rep[U₁](-0.5=>1, 0.5=>1)
    vacuum = oneunit(ph)
    σ₊ = zeros(vacuum ⊗ ph ← Rep[U₁](1=>1) ⊗ ph)
    copy!(block(σ₊, Irrep[U₁](0.5)), ones(1, 1))
    σ₋ = zeros(vacuum ⊗ ph ← Rep[U₁](-1=>1) ⊗ ph)
    copy!(block(σ₋, Irrep[U₁](-0.5)), ones(1, 1))
    σz = ones(ph ← ph)
    copy!(block(σz, Irrep[U₁](-0.5)), -ones(1, 1))
    return Dict("+"=>σ₊, "-"=>σ₋, "z"=>σz)
end

function finite_xxz()
    L = 100
    p = spin_site_ops_u1x() 
    sp, sm, sz = p["+"], p["-"], p["z"]

    pspace = physical_space(sp)

    physpaces = [pspace for i in 1:L]

    hz = 0.7
    Jzz = 1.3
    terms = []
    for i in 1:L
        push!(terms, QTerm(i=>sz, coeff=hz))
    end
    for i in 1:L-1
        push!(terms, QTerm(i=>sp, i+1=>sp', coeff=2))
        push!(terms, QTerm(i=>sm, i+1=>sm', coeff=2))
        push!(terms, QTerm(i=>sz, i+1=>sz, coeff=Jzz))
    end
    ham = QuantumOperator(physpaces, terms)

    initial_state = [-0.5 for i in 1:L]
    for i in 2:2:L
        initial_state[i] = 0.5
    end

    state = prodmps(ComplexF64, physpaces, initial_state)
    observer = QTerm(49=>sp, 50=>sp')

    obs = [expectation(observer, state)]

    circuit = trotter_propagator(ham, -0.01im, stepsize=0.01, order=2)

    for i in 1:10
        apply!(circuit, state)
        push!(obs, expectation(observer, state))
    end

    return obs
end

function infinite_xxz()
    p = spin_site_ops_u1x() 
    sp, sm, sz = p["+"], p["-"], p["z"]
    pspace = physical_space(sp)
    
    hz = 0.7
    Jzz = 1.3
    ham = InfiniteQuantumOperator([pspace])
    push!(ham, QTerm(1=>sz, coeff=hz))
    t = QTerm(1=>sp, 2=>sp', coeff=2)
    push!(ham, t)
    push!(ham, t')
    push!(ham, QTerm(1=>sz, 2=>sz, coeff=Jzz))

    initial_state = [-0.5, 0.5]
    state = prodimps(ComplexF64, [pspace for i in 1:length(initial_state)], initial_state)
    canonicalize!(state)

    observer = QTerm(1=>sp, 2=>sp')
    obs = [expectation(observer, state)]

    circuit = trotter_propagator(ham, -0.01im, unitcellsize=length(state), stepsize=0.01, order=2)

    for i in 1:10
        apply!(circuit, state)
        push!(obs, expectation(observer, state))
    end

    return obs
end

function infinite_xxz_mpo()
    p = spin_site_ops_u1x() 
    sp, sm, sz = p["+"], p["-"], p["z"]
    pspace = physical_space(sp)

    hz = 0.7
    Jzz = 1.3

    m = fromABCD(C=[2*sp, 2*sm, Jzz*sz], B= [sp', sm', sz], D=hz*sz)
    h = MPOHamiltonian([m])
    U = timeevompo(h, -0.01im, WII())

    mpo = InfiniteMPO(U)

    initial_state = [-0.5, 0.5]
    state = prodimps(ComplexF64, [pspace for i in 1:length(initial_state)], initial_state)
    orth = Orthogonalize(trunc=truncdimcutoff(D=200, ϵ=1.0e-8, add_back=0), normalize=true)
    canonicalize!(state, alg=orth)  

    observer = QTerm(1=>sp, 2=>sp')
    obs = [expectation(observer, state)]

    for i in 1:10
        state = mpo * state
        canonicalize!(state, alg=orth)
        push!(obs, expectation(observer, state))
    end 
    return obs
end

function infinite_xxz_mpo2()
    p = spin_site_ops_u1x() 
    sp, sm, sz = p["+"], p["-"], p["z"]
    pspace = physical_space(sp)

    hz = 0.7
    Jzz = 1.3

    m = fromABCD(C=[2*sp, 2*sm, Jzz*sz], B= [sp', sm', sz], D=hz*sz)
    h = MPOHamiltonian([m])

    dt = -0.01im
    dt1, dt2 = complex_stepper(dt)
    U1, U2 = timeevompo(h, dt1, WII()), timeevompo(h, dt2, WII())

    mpo1, mpo2 = InfiniteMPO(U1), InfiniteMPO(U2)

    initial_state = [-0.5, 0.5]
    state = prodimps(ComplexF64, [pspace for i in 1:length(initial_state)], initial_state)
    orth = Orthogonalize(trunc=truncdimcutoff(D=200, ϵ=1.0e-8, add_back=0), normalize=true)
    canonicalize!(state, alg=orth)  

    observer = QTerm(1=>sp, 2=>sp')
    obs = [expectation(observer, state)]

    for i in 1:10
        state = mpo1 * state
        canonicalize!(state, alg=orth)
        state = mpo2 * state
        canonicalize!(state, alg=orth)
        push!(obs, expectation(observer, state))
    end 
    return obs
end



@testset "iTEBD: comparison with TEBD, 1st and 2nd order TimeEvoMPO" begin
    obs1 = finite_xxz()
    obs2 = infinite_xxz()
    @test maximum(abs, obs1 - obs2) < 1.0e-4
    obs3 = infinite_xxz_mpo()
    @test maximum(abs, obs1 - obs3) < 1.0e-2
    obs4 = infinite_xxz_mpo2()
    @test maximum(abs, obs1 - obs4) < 1.0e-4
end
