# Time-Dependent Schrödinger equation

$$
\ket{\psi ( t)} =\sum _{n=1}^{\infty } c_{n}\exp\left(\frac{-iE_{n} t}{\hbar }\right)\ket{\phi _{n}} ;\ket{\psi _{0}} =\sum _{n=1}^{\infty } c_{n}\ket{\phi _{n}} ;\hat{H}\ket{\phi _{n}} =E_{n}\ket{\phi _{n}}
$$

$$
\Rightarrow \underbrace{\begin{pmatrix}
\bra{\phi _{1}}\ket{\psi _{0}}\\
\bra{\phi _{2}}\ket{\psi _{0}}\\
\vdots \\
\bra{\phi _{n}}\ket{\psi _{0}}
\end{pmatrix}}_{\vec{b}} =\underbrace{\begin{pmatrix}
\bra{\phi _{1}}\ket{\phi _{1}} & \bra{\phi _{1}}\ket{\phi _{2}} & \dotsc  & \bra{\phi _{1}}\ket{\phi _{n}}\\
\left(\bra{\phi _{1}}\ket{\phi _{2}}\right)^{\dagger } & \bra{\phi _{2}}\ket{\phi _{2}} & \dotsc  & \bra{\phi _{2}}\ket{\phi _{n}}\\
 & \vdots  &  & \\
\left(\bra{\phi _{1}}\ket{\phi _{n}}\right)^{\dagger } & \left(\bra{\phi _{2}}\ket{\phi _{n}}\right)^{\dagger } & \dotsc  & \bra{\phi _{n}}\ket{\phi _{n}}
\end{pmatrix}}_{\hat{A} \ \text{donde} \ \bra{\phi _{m}}\ket{\phi _{n}} =\text{sum}\left(\int _{\Omega }[ \phi _{m}( x_{i})]^{*} \phi _{n}( x_{j}) d\Omega \right)}\underbrace{\begin{pmatrix}
c_{1}\\
c_{2}\\
\vdots \\
c_{n}
\end{pmatrix}}_{\vec{x}} \Rightarrow \underbrace{\vec{x} =\hat{A} /\vec{b}}_{ \begin{array}{l}
\text{operador}\\
\text{back\ slash}
\end{array}}
$$

Numéricamente, esta inversión de matriz podremos calcularla utilizando el operador backslash $\displaystyle \vec{x} =\vec{A} /\vec{b}$ en Julia. Además, para el caso en que los autoestados son ortonormales (como sería nuestro caso particular) tendremos que los coeficientes vienen dados simplemente como $\displaystyle c_{n} =\bra{\phi _{i}}\ket{\psi _{0}} =\text{sum}\left(\int _{\Omega }[ \phi _{n}( x_{i})]^{*} \psi _{0}( x_{j}) d\Omega \right)$ sin requerir una inversión de la matriz $\displaystyle A$.

# Chequeo de convergencia del método

Por un lado sabemos que un estado inicial normalizado se descompone en una base de autoestados del Hamiltoniano del sistema como $\displaystyle \ket{\psi ^{( n)}( t=0)} =\sum _{j=1}^{n} c_{j}\ket{\phi _{j}}$ entonces la evolución temporal vendrá dada por la expresión $\displaystyle \ket{\psi ^{( n)}( t)} =\sum _{j=1}^{n} c_{j}\exp\left( -\frac{i}{\hbar } \epsilon _{j} t\right)\ket{\phi _{j}} ;\ \bra{\phi _{i}}\ket{\phi _{j}} =\delta _{ij}$. Ahora bien, el valor $\displaystyle n$ tiene en cuenta que en la sumatoria se han tenido en cuenta solo $\displaystyle n$ autoestados, entonces un criterio de convergencia posible sería comparar qué tanto difiere el estado $\displaystyle \ket{\psi ^{( n)}( t)}$ con el estado $\displaystyle \ket{\psi ^{( n+q)}( t)}$ donde este último cuenta con $\displaystyle q$ autoestados adicionales que aportan a la precisión de los resultados, entonces:

$$
\Large
 \begin{array}{l}
\bra{\psi ^{( n)}( t)}\ket{\psi ^{( n+q)}( t)} =\left[\sum _{j=1}^{n}( c_{j})^{*}\exp\left(\frac{i}{\hbar } \epsilon _{j} t\right)\bra{\phi _{j}}\right]\left[\sum _{k=1}^{( n+q)} c_{k}\exp\left( -\frac{i}{\hbar } \epsilon _{k} t\right)\ket{\phi _{k}}\right]\\
\Rightarrow \bra{\psi ^{( n)}( t)}\ket{\psi ^{( n+q)}( t)} =\sum _{j=1}^{n}( c_{j})^{*} c_{j}
\end{array}
\normalsize
$$

la sumatoria anterior en el caso en que los $\displaystyle q$ autoestados adicionales no aporten mayor precisión a los resultados debería cumplirse que $\displaystyle \ket{\psi ^{( n)}( t)} \approx \ket{\psi ^{( n+q)}( t)}$ entonces $\displaystyle \bra{\psi ^{( n)}( t)}\ket{\psi ^{( n+q)}( t)} \approx 1$. Por lo tanto, se computó para cada simulación la sumatoria anterior y se comparó qué tanto se alejaba de la unidad, de esta forma podríamos saber cuántas autoenergías y autoestados agregar a la evolución temporal para mejorar la precisión de los resultados.


# Implementaciones numéricas

## Descomposición del estado inicial como combinación de autoestados

```julia
"""
    initial_coefficients(initial_wave_function,eigen_states,differential_interior_FE_domain)

# Aim
    - This function computes the coefficients of the linear combination of the eigenstates of the Hamiltonian operator.
    The coefficients are computed by the inner product of the initial wave function and the eigenstates.

# Arguments
    - `initial_wave_function::CellField`: Initial wave function.
    - `eigen_states::Vector{CellField}`: Eigenstates of the Hamiltonian operator.
    - `differential_interior_FE_domain::Gridap.CellData.GenericMeasure`: Measure of the interior of the finite element domain.

# Returns
    - `coefficients::Vector{ComplexF64}`: Coefficients of the linear combination of the eigenstates.
"""
function initial_coefficients(
    initial_wave_function::CellField,
    eigen_states::Vector{CellField},
    differential_interior_FE_domain::Gridap.CellData.GenericMeasure)
    coefficients=zeros(ComplexF64,length(eigen_states))
    Threads.@threads for i in eachindex(eigen_states)
        coefficients[i]=sum(∫(conj(eigen_states[i])*initial_wave_function)*differential_interior_FE_domain)
    end
    return coefficients;
end
```

## Evolución de la función de onda (película)
```julia
"""
    evolution_schrodinger(initial_wave_function,eigen_states,eigen_energies,trial_space,differential_interior_FE_domain,time)

# Aim
    - This function computes the evolution of the wave function in time.
    The wave function is given by the linear combination of the eigenstates of the Hamiltonian operator.
    The coefficients of the linear combination are computed by the inner product of the initial wave function and the eigenstates.
    The time evolution is computed by the Schrödinger equation.  

# Arguments
    - `initial_wave_function::CellField`: Initial wave function.
    - `eigen_states::Vector{CellField}`: Eigenstates of the Hamiltonian operator.
    - `eigen_energies::Vector{ComplexF64}`: Eigenvalues of the Hamiltonian operator.
    - `trial_space::FESpace`: Finite element space where the wave function is defined.
    - `differential_interior_FE_domain::Gridap.CellData.GenericMeasure`: Measure of the interior of the finite element domain.
    - `time::Vector{Float64}`: Time vector.

# Returns
    - `wave_function::Vector{CellField}`: Time evolution of the wave function.
"""
function evolution_schrodinger(
    initial_wave_function::CellField,
    eigen_states::Vector{CellField},
    eigen_energies::Vector{ComplexF64},
    trial_space::FESpace,
    differential_interior_FE_domain::Gridap.CellData.GenericMeasure,
    time::Vector{Float64})
    Planck_constant=1.0
    coeffvec=initial_coefficients(initial_wave_function,eigen_states,differential_interior_FE_domain)
    wave_function=Vector{CellField}(undef,length(time))
    for i in eachindex(time)
        factor = coeffvec[1]
        wave_function[i]=interpolate_everywhere((factor*eigen_states[1]),trial_space)
        for j in 2:length(eigen_energies)
            factor = coeffvec[j]*exp(-im*(1.0/Planck_constant)*time[i]*eigen_energies[j])
            wave_function[i]=interpolate_everywhere((wave_function[i]+factor*eigen_states[j]),trial_space)
        end
        wave_function[i]=interpolate_everywhere(wave_function[i],trial_space)
    end
    return wave_function;
end
```
## Evolución de la función de onda (foto)

```julia
function evolution_schrodinger(
    initial_wave_function::CellField,
    eigen_states::Vector{CellField},
    eigen_energies::Vector{ComplexF64},
    trial_space::FESpace,
    differential_interior_FE_domain::Gridap.CellData.GenericMeasure,
    time::Float64)
    Planck_constant=1.0
    coeffvec=initial_coefficients(initial_wave_function,eigen_states,differential_interior_FE_domain)
    # wave_function=interpolate_everywhere((coeffvec[1]*eigen_states[1]),trial_space)
    wave_function=0.0*eigen_states[1]
    for i in eachindex(eigen_energies)
        factor = coeffvec[i]*exp(-im*(1.0/Planck_constant)*time*eigen_energies[i])
        wave_function=interpolate_everywhere((wave_function+factor*eigen_states[i]),trial_space)
    end
    return wave_function;
end
```

## Evolución de la autocorrelación

La función de GRIDAP `interpolate_everywhere()` es muy costosa computacionalmente y (actualmente) esta forma de resolver la dinámica no es eficiente, sin embargo, una forma de prescindir el uso de esta función sería calcular la evolución de la autocorrelación entre la función de onda a un tiempo dado con la función de onda a tiempo inicial. Matemáticamente sería:

$$
\Large
 \begin{array}{l}
\bra{\psi ( t=0)}\ket{\psi ( t)} =\left[\sum _{m=1}^{\infty }( c_{m})^{*}\bra{\phi _{m}}\right]\left[\sum _{n=1}^{\infty } c_{n}\exp\left(\frac{-iE_{n} t}{\hbar }\right)\ket{\phi _{n}}\right]\\
\Rightarrow \bra{\psi _{0}}\ket{\psi ( t)} =\sum _{m=1}^{\infty }\sum _{n=1}^{\infty }( c_{m})^{*} c_{n}\exp\left(\frac{-iE_{n} t}{\hbar }\right)\bra{\phi _{m}}\ket{\phi _{n}} =\sum _{m=1}^{\infty }\sum _{n=1}^{\infty }( c_{m})^{*} c_{n}\exp\left(\frac{-iE_{n} t}{\hbar }\right) \delta _{mn}\\
\Rightarrow \bra{\psi _{0}}\ket{\psi ( t)} =\sum _{n=1}^{\infty }| c_{n}| ^{2}\exp\left(\frac{-iE_{n} t}{\hbar }\right) \therefore p( t) =\left| \bra{\psi _{0}}\ket{\psi ( t)}\right| ^{2}
\end{array}
$$

```julia
function autocorrelation(
    initial_wave_function::CellField,
    eigen_states::Vector{CellField},
    eigen_energies::Vector{ComplexF64},
    differential_interior_FE_domain::Gridap.CellData.GenericMeasure,
    time::Vector{Float64})
    autocorr=similar(time)
    Planck_constant=1.0
    coeffvec=initial_coefficients(initial_wave_function,eigen_states,differential_interior_FE_domain)
    for i in eachindex(time)
        factor = 0.0
        for j in eachindex(coeffvec)
            factor += abs2(coeffvec[j])*exp(-im*(1.0/Planck_constant)*time[i]*eigen_energies[j])
        end
        autocorr[i] = abs2(factor)
    end
    return autocorr;
end
```

Ejemplo de implementación se puede encontrar en [FEMTISE TUTORIAL](https://github.com/mendzmartin/FEMTISE_TUTORIAL/tree/main)