from experiments import runpsdexperiment as rpsd
import numpy as np

def test_rmse_freq_even():
    """
    Tests that two even frequency bins returns as expected
    """
    f1 = np.arange(0, 5, 1)
    f2 = np.arange(0, 5, 1)
    psd1 = np.zeros(len(f1))
    psd2 = np.zeros(len(f2)) + 10
    ftest, psdtest = rpsd.rmse_frequency_bins(f1, psd1, f2, psd2)
    assert(np.allclose(f1, ftest))
    assert(np.allclose(psdtest, psd2))


def test_rmse_freq_f1_res():
    """
    Tests that with the first frequency bin higher resolution the results are as expected
    """
    f1 = np.arange(0, 5, .5)
    f2 = np.arange(0, 5, 1)
    psd1 = np.arange(0, 5, .5)
    psd2 = np.zeros(len(f2)) + 10
    ftest, psdtest = rpsd.rmse_frequency_bins(f1, psd1, f2, psd2)
    print(ftest)
    print(psdtest)
    assert(np.allclose(f2, ftest))
    assert(np.allclose(psd2 - np.array([0, 1.5/2, 3.5/2, 5.5/2, 7.5/2]), psdtest))

def test_rmse_freq_f2_res():
    """
    Same as above test, but switches f1 and f2/psd1 and psd2
    :return:
    """
    f2 = np.arange(0, 5, .5)
    f1 = np.arange(0, 5, 1)
    psd2 = np.arange(0, 5, .5)
    psd1 = np.zeros(len(f1)) + 10
    ftest, psdtest = rpsd.rmse_frequency_bins(f1, psd1, f2, psd2)
    #print(ftest)
    assert(np.allclose(f1, ftest))
    assert(np.allclose(psd1 - np.array([0, 1.5/2, 3.5/2, 5.5/2, 7.5/2]), psdtest))

def test_rmse_freq_f1_res_nc():
    """
    Tests where the f1 frequency bins have higher resolution, but not at a constant ratio

    """
    f1 = np.arange(0, 5, .4)
    f2 = np.arange(0, 5, 1)
    psd1 = np.arange(0, 5, .4)
    psd2 = np.zeros(len(f2)) + 10
    ftest, psdtest = rpsd.rmse_frequency_bins(f1, psd1, f2, psd2)
    assert (np.allclose(f2, ftest))
    assert (np.allclose(psd2 - np.array([0, .4*(.4 + .8) + (1.2)*.2,  .2*(1.2) + .4*(1.6 + 2), .4*(2.4 + 2.8) + .2*(3.2),
                                         .2*(3.2) + .4*(3.6 + 4)]), psdtest))

def test_rmse_freq_f1_res_nr():
    """
        Tests where the f1 frequency bins have higher resolution, but not rational number.
        Also, result will be different length than original frequencies

        """
    f1 = np.arange(0, 2*np.pi, np.pi/4)
    f2 = np.arange(0, 5, 1)
    psd1 = np.arange(0, 2*np.pi, np.pi/4)
    psd2 = np.zeros(len(f2)) + 10
    ftest, psdtest = rpsd.rmse_frequency_bins(f1, psd1, f2, psd2)
    assert (np.allclose(np.arange(0, 6, 1), ftest))
    assert (np.allclose(
        np.array([10, 10, 10, 10, 10, 0]) - np.array([0, (np.pi/4)*np.pi/4 + (2*np.pi/4)*(1-np.pi/4),
                         (2*np.pi/4)*(2*np.pi/4-1) + (3*np.pi/4)*(2 - 2*np.pi/4),
                         (3*np.pi/4)*(3*np.pi/4 - 2) + (np.pi)*(3 - 3*np.pi/4),
                        (np.pi)*(np.pi - 3) + (5*np.pi/4)*(np.pi/4) + (6*np.pi/4)*(1 - (np.pi/4 + np.pi - 3)),
                        (6*np.pi/4)*(6*np.pi/4 - 4) + (7*np.pi/4)*(5 - 6*np.pi/4)]), psdtest))

