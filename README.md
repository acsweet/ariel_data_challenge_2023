# Ariel Data Challenge 2023
This is the repository for the 3rd place solution in the 2023 <a href="https://www.ariel-datachallenge.space/">Ariel Data Challenge</a>.

## Setup
- download the data from the <a href="https://www.ariel-datachallenge.space/ML/documentation/description">ADC 2023 challenge website</a>, and place it in the <code>data</code> directory
    - check the <code>training_path</code> variables in the notebooks and adjust if needed
- <code>python=3.11.3</code> was used
- install the packages from the <code>requirements.txt</code> file
- for spectral scores and taurex3, please see the notes below (used in final_inference.ipynb)

## Notes
- code in the <code>base_code</code> directory is from the <a href="https://github.com/ucl-exoplanets/ADC2023-baseline">Baseline github repository</a> provided by UCL
- code in <code>tabnet</code> directory is from a <a href="https://github.com/titu1994/tf-TabNet">TabNet for Tensorflow 2.0 repository</a>
- for details on calculating the spectral scores, and setting up <a href="https://github.com/ucl-exoplanets/TauREx3_public">TauREx3</a>, please see the <a href="https://github.com/ucl-exoplanets/ADC2023-baseline#metrics">Metrics</a> section of the baseline repository
