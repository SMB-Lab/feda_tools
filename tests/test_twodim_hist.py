
import pytest

def test_calc_list_entries():
    """
    test to ensure that the calc_list contains the supported calculations
    """
    from feda_tools.twodim_hist import calc_list
    assert "Mean Macro Time (sec)" in calc_list
    assert "Sg/Sr (prompt)" in calc_list
    assert "S(prompt)/S(total)" in calc_list

def test_parse_args(monkeypatch):
    """
    Test to make sure that parse_args returns the expected arguments
    """
    from feda_tools.twodim_hist import parse_args
    import feda_tools.twodim_hist

    # the mock data is designed to pass the arg_check, so just mock arg_check 
    # and return it.
    def mock_arg_check(arg):
        return arg
    
    monkeypatch.setattr(feda_tools.twodim_hist, "arg_check", mock_arg_check)

    mock_folder = 'data_folder'
    mock_file = 'plot_file.yaml'
    data_folder, plot_file = parse_args(['data_folder', 'plot_file.yaml',])

    assert data_folder == mock_folder
    assert plot_file == mock_file

def test_arg_check_exception(monkeypatch):
    """
    Test to ensure that arg_check will raise an exception if provided a path 
    that does not exist.
    """
    import os.path
    from feda_tools.twodim_hist import arg_check
    from argparse import ArgumentTypeError

    path = "mock_path"
    def mock_exists(path):
        return False

    monkeypatch.setattr(os.path, 'exists', mock_exists)
   
    errmsg = str(path + ' could not be found. ' 
                 + 'Check for typos or for errors in your relative path string')
   
    with pytest.raises(ArgumentTypeError) as excinfo:
        arg_check(path)
    assert str(excinfo.value == errmsg)

def test_arg_check_pass(monkeypatch):
    """
    Test to ensure that arg_check will pass and return the path when provided 
    a path that exists
    """
    import os.path
    from feda_tools.twodim_hist import arg_check

    mock_path = "mock_path"
    def mock_exists(mock_path):
        return mock_path

    monkeypatch.setattr(os.path, 'exists', mock_exists)

    assert mock_path == arg_check(mock_path)

def test_get_plot_dict(monkeypatch):
    """
    test to ensure that the get_plot_dict method returns a dict with the 
    proper format.
    """
    from feda_tools.twodim_hist import get_plot_dict
    import yaml


    def mock_safe_load(yaml_file):
        mock_dict = {
            'plot1': 
            {
                'xlabel': 'Number of Photons', 
                'xfolder': 'bi4_bur', 
                'ylabel': 'Duration (ms)', 
                'yfolder': 'bi4_bur'
            }
        }
        return mock_dict
        

    monkeypatch.setattr(yaml, "safe_load", mock_safe_load)

    mock_plot_file = "plot.yaml"
    plot_dict = get_plot_dict(mock_plot_file)
    mock_dict = mock_safe_load(mock_plot_file)
    assert plot_dict == mock_dict
    
def test_get_data():
    """
    Ensure that get_data returns the proper dataframe using real test data.
    
    test_data.bur was created by running get_data on the actual data path to 
    obtain a dataframe that represented the test data in full. The test data 
    was then written back to csv in combined form with index=False using the
    DataFrame.to_csv method.
    
    """
    import pandas as pd
    import numpy as np
    from feda_tools.twodim_hist import get_data

    expected_data_path= './tests/test data/test_get_data/test_data.bur'
    expected_df = pd.read_csv(expected_data_path, sep = '\t')
    print(expected_df)
    
    actual_data_path= './tests/test data/High_1hr_split/burstwise_All 0.1771#60_0/bi4_bur/'
    actual_df = get_data(actual_data_path)

    assert np.array_equal(actual_df, expected_df)

def test_get_calc_MMT_secs(monkeypatch):
    """
    Test that a "Mean Macro Time (secs)" (MMT) calculation is handled
    correctly by returning a pandas series whose values have been divided
    by 1000
    """
    import pandas as pd
    import feda_tools.twodim_hist
    from feda_tools.twodim_hist import get_calc
    import numpy as np

    mock_folder = 'mock_folder'
    def mock_get_data(mock_folder):
        data = {"Mean Macro Time (ms)" : [1.0, 2.0, 3.0]}
        mock_df = pd.DataFrame(data)
        return mock_df
    
    monkeypatch.setattr(feda_tools.twodim_hist, "get_data", mock_get_data)

    expected_df = pd.DataFrame({"Mean Macro Time (sec)" : [0.001, 0.002, 0.003]}) 
    actual_df = get_calc("Mean Macro Time (sec)", mock_folder)

    assert np.array_equal(actual_df, expected_df)






