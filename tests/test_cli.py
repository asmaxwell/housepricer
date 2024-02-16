"""
Created by Andy S Maxwell 14/02/2024
Test for cli interface for housepricer
"""
import pytest
from os.path import exists
from housepricer import cli

@pytest.mark.slow
def test_full_run_evolution():
    directory = "data/"
    cli.full_run_evolution(directory, directory, "data/codepo_gb/Data/CSV/", 5, 3)
    assert(exists(directory + "EvolveModel.save"))
    return


@pytest.mark.slow
def test_full_run_random():
    directory = "data/"
    cli.full_run_random(directory, directory, "data/codepo_gb/Data/CSV/", 10)
    assert(exists(directory + "RandomModel.save"))
    return

@pytest.mark.slow
def test_cal_run_evolution():
    directory = "data/"
    cli.cal_run_evolution(directory, 5, 3)
    assert(exists(directory + "EvolveModel.save"))
    return


@pytest.mark.slow
def test_cal_run_random():
    directory = "data/"
    cli.cal_run_random(directory, 10)
    assert(exists(directory + "RandomModel.save"))
    return
