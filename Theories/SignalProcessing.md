# Signal Processing
## Created: 2025.01.10
## Signal
Signal: 
- functions of independent variables that carry information on the nature of a physical phenomenon
- assocaited with a system generating or extracting the information
- acoustic signals: voice, audio
- electric signals: voltage or current
- video signals

Classification
- Continuous Time Signal
- Discrete Time Signal
  - domain of discrete time instants
  - but its value in a continuous range 
- Digital Signal
- Even or odd signal
- deterministic and stochastic
- Energy 
  - have finite total energy and zero average power
  - $$E_\infty = lim \int_T{x^2(t)dt}$$
- Power signal
  - have finite total energy and finite average power
  - $$P_\infty = lim {1\over T}\int^{T\over 2}_{-T\over 2}|x(t)|^2dt$$
- periodic and non-periodic signal
## Convolution and Correlation
**Convolution Integral**        
$$y(t) = x(t)*h(t) = \int^\infty _{-\infty} x(\tau)h(t-\tau)d\tau$$         
- basic tool for analyzing linear systems in time domain
- closely related to the response of a linear system

**Correlation**     
- $$x(t) = y(t)$$, auto-correlation
- $$x(t) \neq y(t)$$, cross-correlation, similarity between two signals

$$y(t) = x(t)*h(t) = \int^\infty _{-\infty} x(\tau)h(t-\tau)d\tau$$     
    
**Elementary CT signals**
- step signal
- ramp signal
- rectangular and triangular pulses
- impulse(Dirac delta) signal
  - Sifting property $$\int^{\infty}_{-\infty}f(t)\delta(t-t_0)dt = f(t_0)$$
  - Shifting property $$f(t)*\delta(t-t_0)= \int^{\infty}_{-\infty}f(t-\tau)\delta(\tau-t_0)d\tau = f(t-t_0)$$
  - Derivative property
- sinc signal
- exponential signal
- complex exponent


## Complex exponent
Euler Formula   
$$e^{jx}= cos(x)+jsin(x)$$  
$$e^{-jx}= cos(x)-jsin(x)$$  

## Fourier Series
- A signal is periodic if $$x(t)= x(t+T)$$
- the signal repeats every T seconds
- Period: the time in which the sinusoidal signal completes its one cycle
- Frequency: the number of cycles in one period
  - angular frequency $$\omega_0 [rad/s]$$
  - frequency $$f=1/T [Hz]$$
- Composite sinusoid
  - $$x(t)=x_1(t)+x_2(t)$$ is periodic if and only if their frequencies have common divisor
  - Fundamental frequency: greatest common divisor of $$\omega_1$$ and $$\omega_2$$

## Fourier Transform

## Laplace Transform