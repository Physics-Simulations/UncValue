# Real use examples of Uncertainty-Value package.

In this Jupyter Notebook a real physics-lab practice will be solved with the support of UncValue package. The practice was part of the Physics degree at Autonomous University of Barcelona in 2017.

## Heat transport in a cylindrical bar

The main purpose of this practice is to study the variation of the temperature along different metalic cylindrical bars. In this situation, heat is basically trasnported by conduction, so that the heat trasnport can be computed by means of Fourier equation. Considering a cylindrical bar of length ![equation](https://latex.codecogs.com/gif.latex?L) the solution to the Fourier equation with boundary conditions 

![equation](https://latex.codecogs.com/gif.latex?\theta(x=0)=\theta_0) 

![equation](https://latex.codecogs.com/gif.latex?\theta(x=L)=0) 

is given by

![equation](https://latex.codecogs.com/gif.latex?\theta(x)=\theta_0e^{-px})

where 

![equation](https://latex.codecogs.com/gif.latex?\theta(x)=T(x)-T_e)

![equation](https://latex.codecogs.com/gif.latex?p=\sqrt{2&space;\lambda&space;/&space;K&space;r})

- ![equation](https://latex.codecogs.com/gif.latex?T_e) : Environmental temperature
- ![equation](https://latex.codecogs.com/gif.latex?\lambda) : Lateral losses coefficient
![equation](https://latex.codecogs.com/gif.latex?K) : Thermal conductivity
- ![equation](https://latex.codecogs.com/gif.latex?r) : Bar radius

## Result to be obtained

Besides qualitatively studying the temperature propagation along the bars, which can be plotted, the thermal conductivity can be quantitatively determined. The lateral losses coefficient must be equal for all the bars, regardless of the material, as it only depends on the object geometry. In this practice, the bars have the same geometry and dimensions, so that the conductivity of the different materials that form the bars (Aluminium, Laton and Iron) can be computed by using the fact that the lateral losses coefficients are equal.
