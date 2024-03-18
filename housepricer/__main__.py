#!/usr/bin/env python3

import argparse
import cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="housepricer"
                                , description="Train models for predicting house prices"
                                , epilog="Thanks for using %(prog)s! :)")
    
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--hyperparameter_search", default="evolve")
    parser.add_argument("--load_cal", default="False")
    parser.add_argument("--model_type", default="random")
    parser.add_argument("--population", default=None)
    parser.add_argument("--generations", default=None)
    parser.add_argument("--iterations", default=None)
    args = parser.parse_args()   
    postcode_directory = "data/codepo_gb/Data/CSV/"

    data_directory = args.data_dir
    if args.model_dir == None:
        model_directory = data_directory
    else:
        model_directory = args.model_dir

    if args.hyperparameter_search == "random":
        if args.iterations == None:
            print("Error need --iterations flag with --hyperparameter_search random")
        else:
            if args.load_cal == "False":
                cli.full_run_random(data_directory
                                    , model_directory
                                    , postcode_directory
                                    , int(args.iterations)
                                    , args.model_type )
            else:
                cli.cal_run_random(model_directory, int(args.iterations), args.model_type )
    
    else:
        if args.population == None or args.generations == None:
            print("Error need --population and --generations flags with --hyperparameter_search evolve")
        else:
            if args.load_cal == "False":
                cli.full_run_evolution(data_directory
                                       , model_directory
                                       , postcode_directory
                                       , int(args.population)
                                       , int(args.generations)
                                       , args.model_type )
            else:
                cli.cal_run_evolution(model_directory, int(args.population), int(args.generations), args.model_type )




    
    


   

