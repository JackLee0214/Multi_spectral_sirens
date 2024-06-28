
## `Multi_spectral_sirens`

These files include the codes and data to re-produce the results of the work  _Multi-spectral sirens: Gravitational-wave cosmology with (multi-) sub-populations of binary black holes_, arXiv: [2406.11607](https://arxiv.org/abs/2406.11607)
 [Yin-Jie Li](https://inspirehep.net/authors/1838354) ,  [Shao-Peng Tang](https://inspirehep.net/authors/1838355) ,  [Yuan-Zhu Wang](https://inspirehep.net/authors/1664025),  [Yi-Zhong Fan](https://inspirehep.net/authors/1040745)

#### Main requirements
- [BILBY](https://git.ligo.org/lscsoft/bilby)
- [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html)
- [ICAROGW](https://anaconda.org/simone.mastrogiovanni/icarogw/2021.11.04.101404/download/env_creator.yml)

#### Data
The events posterior samples are adopted from the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/eventapi/html/GWTC/), here samples of events with SNR > 11 are used for analysis and stored in `data/BBH_snr_11_Nobs_42.pickle`. 

The injection campaign `data/O1_O2_O3_det_frame_SNR9.inj`
is adopted from [Abbot et al.](https://zenodo.org/records/5645777/files/O1_O2_O3_det_frame_SNR9.inj)
  
#### Hierarchical Bayesian inference
- Run the python script `Multispectral.py` , and specify the TwoSpin model, PP model and PS model by setting `label='pls_twospin'`, `label='pp'`, and `label='pls'` in the script.


The inferred results `*.json` will be saved to `results`

#### Results
- `pp.pickle` is the posterior samples inferred by the pp model.
- `pls.pickle` is the posterior samples inferred by the pls model.
- `pls_twospin.pickle` is the posterior samples inferred by the twospin model.


#### Generate figures
Run the python script `Figure_script.py`

The figures will be saved to `figures`
  
#### Acknowledgements
The  [FigureScript](https://dcc.ligo.org/public/0171/P2000434/003/Produce-Figures.ipynb) from [LIGO Document P2000434](https://dcc.ligo.org/LIGO-P2000434/public) is referenced to produced figures in this work.


  


