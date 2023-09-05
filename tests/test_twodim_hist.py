
def test_calc_list_entries():
    """
    test to ensure that the calc_list contains the supported calculations
    """
    from feda_tools.twodim_hist import calc_list
    assert "Mean Macro Time (sec)" in calc_list
    assert "Sg/Sr (prompt)" in calc_list
    assert "S(prompt)/S(total)" in calc_list

