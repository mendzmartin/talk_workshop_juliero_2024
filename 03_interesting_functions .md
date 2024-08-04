# Funciones interesantes disponibles en `GRIDAP` y `FEMTISE`

## Construcción de funciones arbitrarias

```julia
function arbitrary_function_Gridap(arbitrary_function::Function,params::Tuple,FE_domain)
    return CellField(x->arbitrary_function(x,params),FE_domain);
end
```

## Integración de funciones arbitrarias

```julia
function integration_gridap(arbitrary_function_Gridap::CellField,diff_FE_domain::Gridap.CellData.GenericMeasure)
    return sum(integrate(arbitrary_function_Gridap,diff_FE_domain)); # ejemplo
end
```


```julia
function integration_gridap(arbitrary_function_Gridap::Vector{CellField},diff_FE_domain::Gridap.CellData.GenericMeasure,TrialSpace::FESpace)
    multifield_arbitrary_function_Gridap=interpolate_everywhere(arbitrary_function_Gridap,TrialSpace);
    f1,f2=f;
    return sum(∫(f1'*f2)*dΩ)+sum(∫(f2'*f1)*dΩ); # ejemplo
end
```

## Funciones interesantes disponibles en FEMTISE

[$L_2$ norm](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/useful_functions_for_FEM_objects.jl#L127-L139)

$$
\text{norm}_{L_{2}} =\left(\int \psi ^{*} \psi d\Omega \right)^{1/2}
$$

[Total probability density](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/useful_functions_for_FEM_objects.jl#L238-L249):

$$
    \rho (\vec{x}) = \Vert \psi (\vec{x})\Vert ^{2} =\int [ \psi (\vec{x})]^{*} \psi (\vec{x}) d\Omega
$$
and if $\rho (\vec{x}) :=\rho ( x_{1} ,x_{2})$ then [the reduced probability density](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/useful_functions_for_FEM_objects.jl#L258-L275) will be
$$
    \rho ( x_{1}) =\int [ \psi ( x_{1} ,x_{2})]^{*} \psi ( x_{1} ,x_{2}) dx_{2}
$$

Differential [Shannon](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/useful_functions_for_FEM_objects.jl#L288-L301) and [Rènyi](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/useful_functions_for_FEM_objects.jl#L313-L327) entropies and Mutual information
$$
    S_{\text{Sh}} = -\int \rho (\vec{x})\log_{2}[ \rho (\vec{x})] d\Omega \\
    S_{\text{Rè}}( \alpha ) = \frac{1}{( 1-\alpha )}\log_{2}\left\{\int [ \rho (\vec{x})]^{\alpha } d\Omega \right\}
$$
where $0< \alpha \in \mathbb{R} < 1$.
Si $\rho (\vec{x}) =\rho ( x_{1} ,x_{2})$ entonces [la información mutua](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/useful_functions_for_FEM_objects.jl#L416-L429) vendrá dada por:
$$
    \rho ( x_{i}) =\int \rho ( x_{i} ,x_{j}) dx_{j}; S_{\text{Sh}}^{x_{i}} =-\int \rho ( x_{i})\log_{2}[ \rho ( x_{i})] dx_{i} \\
    I_{\text{Sh}} =\int \rho ( x_{1} ,x_{2})\log_{2}\left[\frac{\rho ( x_{1} ,x_{2})}{\rho ( x_{1}) \rho ( x_{2})}\right] d\Omega  =\left( S_{\text{Sh}}^{x_{1}} +S_{\text{Sh}}^{x_{2}} -S_{\text{Sh}}\right)
$$

[Expectation value](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/useful_functions_for_FEM_objects.jl#L470-L485) and [dispersion](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/useful_functions_for_FEM_objects.jl#L525-L540).
$$
    \left\langle f^{n} \right\rangle =\left\langle \psi(\vec{x}) | [f(x)]^n | \psi(\vec{x}) \right\rangle =\int [ \psi (\vec{x})]^{*} [f( x)]^n \psi (\vec{x}) d\Omega \\
    \sigma _{f} =\sqrt{\left\langle f^{2}\right\rangle -\left\langle f \right\rangle ^{2}}
$$

## Algunos problemas solucionados

### Cómo integramos funciones elementales aplicadas a objetos de Elementos Finitos?

```julia
f::Function, u::CellField
sum(∫(f∘u)dΩ)
```

### Cómo hacemos para realizar integrales reducidas dentro de GRIDAP? (no esta realmente preparado para integración parcial)

[DataInterpolations.jl](https://docs.sciml.ai/DataInterpolations/stable/): DataInterpolations.jl is a library for performing interpolations of one-dimensional data. Interpolations are a very important component of many modeling workflows. Often, sampled or measured inputs need to be transformed into continuous functions or smooth curves for simulation purposes. In many scientific machine learning workflows, interpolating data is essential to learn continuous models. DataInterpolations.jl can be used for facilitating these types of workflows. By "data interpolations" we mean techniques for interpolating possibly noisy data, and thus some methods are mixtures of regressions with interpolations (i.e. do not hit the data points exactly, smoothing out the lines).

```julia
using DataInterpolations
function gridap_interpolation_function(x,x_component,fx_vector,x_vector)
    f=interpolation_function(fx_vector,x_vector)
    return f(x[x_component])
end

function interpolation_function(fx_vector,x_vector;TypeInterpolation=AkimaInterpolation)
    f = TypeInterpolation(fx_vector,x_vector)
    return f
end
```

Entonces podemos leer resultados de grillas 2D, crear nuevas grillas 1D y por medio de las interpolaciones anteriores integrar (usando la semántica de Gridap) en estos nuevos dominios. Otra opción sería integrar directamente en los dominios 2D haciendo uso de funciones aproximadas de tipo Delta de Dirac (de echo esto es lo que usa en FEMTISE.jl).

### Cómo se hizo para considerar diferentes masas en los problemas bidimensionales?

[Fomulaciones de tipo Sturm-Liouville](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/miscellaneous_functions.jl#L42-L53)

### Cómo hacemos para leer datos archivos jld2 y continuar con simulaciones?

[JLD2.jl](https://github.com/JuliaIO/JLD2.jl): JLD2 saves and loads Julia data structures in a format comprising a subset of HDF5, without any dependency on the HDF5 C library. JLD2 is able to read most HDF5 files created by other HDF5 implementations supporting HDF5 File Format Specification Version 3.0 (i.e. libhdf5 1.10 or later) and similarly those should be able to read the files that JLD2 produces. JLD2 provides read-only support for files created with the JLD package.

```julia
using JLD2
```

[charge_results 1D](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/post_proccesing_data_function.jl#L103-L111)
[charge_results 2D](https://github.com/mendzmartin/FEMTISE.jl/blob/c4c72d603e9e8516f08a37f966d3ee3b91e7f719/src/functions/post_proccesing_data_function.jl#L153-L161)

> Es muy útil preguntar e interactuar con la comunidad de Gridap para resolver problemas [https://gitter.im/Gridap-jl/community](https://gitter.im/Gridap-jl/community) y estudiar en detalle los tutoriales [https://github.com/gridap/Tutorials](https://github.com/gridap/Tutorials).