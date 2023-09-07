
def test_calc_list_entries():
    """
    test to ensure that the calc_list contains the supported calculations
    """
    from feda_tools.twodim_hist import calc_list
    assert "Mean Macro Time (sec)" in calc_list
    assert "Sg/Sr (prompt)" in calc_list
    assert "S(prompt)/S(total)" in calc_list

def test_parse_args():
    """
    Test to make sure that parse_args returns the expected arguments
    """
    from feda_tools.twodim_hist import parse_args

    mock_folder = 'data_folder'
    mock_file = 'plot_file.yaml'
    data_folder, plot_file = parse_args(['data_folder', 'plot_file.yaml',])

    assert data_folder == mock_folder
    assert plot_file == mock_file

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
    
