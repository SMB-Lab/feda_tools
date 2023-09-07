
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
    
