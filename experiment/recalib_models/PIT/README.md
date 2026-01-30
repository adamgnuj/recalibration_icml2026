Perform PIT recalibration with using the _CDF of the
PIT transform_ (which is a $[0,1] \to [0,1]$ function) computed on the calibration set, as a transformation on the predicted CDF values on the test set.

Since a random variable substituted into its own CDF is $U[0,1]$, this simple approach produces PIT calibrated predictions, such as the isotonic regression approach of [Kuleshov et al., 2018, Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://arxiv.org/abs/1807.00263).