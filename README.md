# muTOV
Microsecond Machine learning TOV emulator for Gravitational wave astrophysics 

## Installation instructions
```shell
git clone https://github.com/spencermagnall/muTOV.git
cd muTOV
pip install -r requirements.txt
pip install .
```
## ML model training and benchmarking
SIMON PLEASE FILL THIS IN! 

## Inference guide
First we import `muTOV` and `bilby`

```python
from muTOV import source
import bilby
```


Parameter estimation on Binary Neutron star signals can be performed in the usual way with
bilby by replacing the standard frequency domain source model `bilby.gw.source.lal_binary_neutron_star`
with `muTOV.source.mutov_lal_binary_neutron_star_H_0`. 

E.g: 
```python
# Create the waveform_generator using the muTOV frequency domain source model
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=mutov_lal_binary_neutron_star_H_0,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)
```

You will also need to define priors for the EOS parameters (currently only Piecewise polytrope is supported) and Hubble's constant 
```python
# Set EOS priors
priors["Gamma_1"] = bilby.core.prior.Uniform(2.0,4.5,name="gamma_1")
priors["Gamma_2"] = bilby.core.prior.Uniform(1.5,4.5,name="gamma_2")
priors["Gamma_3"] = bilby.core.prior.Uniform(1.5,4.5,name="gamma_3")
priors["log_P1"] = bilby.core.prior.Uniform(33.5,34.8,name="log_P1")
priors["H_0"] = bilby.core.prior.Uniform(40,140,name="H_0")
```
If you are not interested in inferring cosmology then you can fix H_0 at your chosen cosmology.
Note that we use the approximation

$`H_0 = \frac{cz}{D_L},`$

which is valid for small redshifts. 

Further details and an example may be found in `bns_example.py` in the examples folder 

## EOS and $`H_0`$ stacking

TODO - Spencer

## Citation guide
If you use `muTOV` for a scientific publication, please cite
* https://arxiv.org/abs/2410.07754


We also use the following packages that you should consider citing if you use them:
* [Bilby](https://git.ligo.org/lscsoft/bilby/-/tree/master)
* [LALSuite](https://lscsoft.docs.ligo.org/lalsuite/)
* [Scikit-learn](https://scikit-learn.org/stable/index.html)
* [Dynesty](https://dynesty.readthedocs.io/en/latest/)
* [Nestle](http://kylebarbary.com/nestle/)


