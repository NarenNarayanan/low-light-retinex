# NIQE model parameters

`niqe_pris_params.npz` holds the pre-trained "pristine" multivariate Gaussian model used by the NIQE
metric (36-dimensional mean, 36x36 covariance, and the 7x7 Gaussian window), fitted on pristine natural
images by the authors of:

A. Mittal, R. Soundararajan and A. C. Bovik, "Making a 'completely blind' image quality analyzer,"
IEEE Signal Processing Letters, vol. 20, no. 3, pp. 209-212, 2013.

The file was obtained from the BasicSR project (`basicsr/metrics/niqe_pris_params.npz`,
https://github.com/XPixelGroup/BasicSR), which converted the parameters from the authors' public MATLAB
release. Please check the terms of the original release and of BasicSR before redistributing this file
outside an academic setting.
