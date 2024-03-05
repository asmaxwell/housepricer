"""
Created by Andy S Maxwell 14/02/2024
Test for cli interface for housepricer
"""
import pytest
import logging
logging.getLogger('tensorflow').disabled = True 
from os.path import exists
from housepricer import cli

@pytest.mark.slow
def test_full_run_evolution():
    directory = "data/"
    cli.full_run_evolution(directory, directory, "data/codepo_gb/Data/CSV/", 3, 2)
    assert(exists(directory + "EvolveModel.save"))
    return


@pytest.mark.slow
def test_full_run_random():
    directory = "data/"
    cli.full_run_random(directory, directory, "data/codepo_gb/Data/CSV/", 5)
    assert(exists(directory + "RandomModel.save"))
    return

@pytest.mark.slow
def test_full_hist_run_random():
    directory = "data/"
    cli.full_run_random(directory, directory, "data/codepo_gb/Data/CSV/", 5, "notrandom")
    assert(exists(directory + "RandomModel.save"))
    return

@pytest.mark.slow
def test_cal_run_evolution():
    directory = "data/"
    cli.cal_run_evolution(directory, 3, 2)
    assert(exists(directory + "EvolveModel.save"))
    return


@pytest.mark.slow
def test_cal_run_random():
    directory = "data/"
    cli.cal_run_random(directory, 5)
    assert(exists(directory + "RandomModel.save"))
    return

@pytest.mark.slow
def test_cal_hist_run_random():
    directory = "data/"
    cli.cal_run_random(directory, 5, "notrandom")
    assert(exists(directory + "RandomModel.save"))
    return
