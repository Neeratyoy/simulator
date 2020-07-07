import sys
import argparse
if '..' not in sys.path:
    sys.path.append('..')

import pandas as pd
import numpy as np
import networkx as nx
import copy
import scipy as sp
import math
import seaborn
import pickle
import warnings
import matplotlib
import re
import multiprocessing
import torch

from botorch import fit_gpytorch_model
from botorch.exceptions import BadInitialCandidatesWarning
import botorch.utils.transforms as transforms
from lib.inference import make_bayes_opt_functions, pdict_to_parr, parr_to_pdict, CalibrationLogger, save_state, load_state, gen_initial_seeds
from lib.inference_kg import qKnowledgeGradient
import time, pprint

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from lib.mobilitysim import MobilitySimulator
from lib.dynamics import DiseaseModel
from bayes_opt import BayesianOptimization
from lib.parallel import *
from lib.distributions import CovidDistributions
from lib.plot import Plotter

from lib.mobilitysim import MobilitySimulator
from lib.calibrate_parser import make_calibration_parser

sys.path.append('../../DEHB/')
from dehb import DE, AsyncDE

if __name__ == '__main__':

    '''
    Command line arguments
    '''

    parser = make_calibration_parser()
    args = parser.parse_args()
    seed = args.seed or 0
    args.filename = args.filename or f'calibration_{seed}'

    # check required settings
    if not (args.mob and args.area and args.country and args.start and args.end):
        print(
            "The following keyword arguments are required, for example as follows:\n"
            "python calibrate.py \n"
            "   --country \"GER\" \n"
            "   --area \"TU\" \n"
            "   --mob \"lib/tu_settings_10.pk\" \n"
            "   --start \"2020-03-10\" \n"
            "   --end \"2020-03-26\" \n"
        )
        exit(0)

    '''
    Genereate essential functions for Bayesian optimization
    '''

    (objective,
     generate_initial_observations,
     initialize_model,
     # optimize_acqf_and_get_observation,
     eval_objective,
     case_diff,
     unnormalize_theta,
     header) = make_de_opt_functions(args=args)

    # logger
    logger = CalibrationLogger(
        filename=args.filename,
        measures_optimized=args.measures_optimized,
        verbose=not args.not_verbose)

    # generate initial training data (either load or simulate)
    if args.load:

        # load initial observations
        state = load_state(args.load)
        train_theta = state['train_theta']
        train_G = state['train_G']
        train_G_sem = state['train_G_sem']
        best_observed_obj = state['best_observed_obj']
        best_observed_idx = state['best_observed_idx']

        header.append('')
        header.append('Loaded initial observations from: ' + args.load)
        header.append(f'Observations: {train_theta.shape[0]}, Best objective: {best_observed_obj}')

        # write header and best prior observations
        logger.log_initial_lines(header)
        for i in range(train_theta.shape[0]):
            loaded_train_G_objectives = objective(train_G[:i+1])
            loaded_best_observed_obj = loaded_train_G_objectives[
                loaded_train_G_objectives.argmax()].item()
            logger.log(
                i=i - train_theta.shape[0],
                time=0.0,
                best=loaded_best_observed_obj,
                case_diff=case_diff(train_G[i]),
                objective=objective(train_G[i]).item(),
                theta=unnormalize_theta(train_theta[i].squeeze())
            )

    else:

        # write header
        logger.log_initial_lines(header)

        # generate initial training data
        train_theta, train_G, train_G_sem, best_observed_obj, best_observed_idx = generate_initial_observations(
            n=args.ninit, logger=logger)
        # print("\nSHAPE OF TRAIN_THETA: {}\n".format(train_theta.shape))
    # init model based on initial observations
    # mll, model = initialize_model(train_theta, train_G, train_G_sem)
    # DE-CHANGE: creating and initializing DE model
    args.pop_size = args.ninit
    de = AsyncDE(dimensions=train_theta.shape[1], pop_size=args.pop_size,
                 mutation_factor=args.mutation, crossover_prob=args.crossover,
                 strategy=args.strategy, async_strategy=args.de_type)
    de.population = train_theta.numpy()
    de.fitness = train_G.numpy()
    de.inc_score = best_observed_obj
    # print("\nDE SETUP done: {}, {}, {}\n".format(de.population, de.fitness, de.f_objective))

    best_observed = []
    best_observed.append(best_observed_obj)

    # run n_iterations rounds of Bayesian optimization after the initial random batch
    # for tt in range(args.niters):
    # DE-CHANGE: one generation of DE constitutes pop_size number of function evaluations
    n_iters = args.niters % args.ninit
    remaining_iters = args.niters - n_iters * args.ninit
    for tt in range(args.niters):

        t0 = time.time()

        parent_idx = tt % de.pop_size

        # DE-CHANGE: no model-fitting, acquisition function
        target = de.population[parent_idx]
        donor = de.mutation(current=target)
        trial = de.crossover(target, donor)
        new_theta = torch.tensor(trial, dtype=torch.float)
        new_G, new_G_sem = eval_objective(new_theta.squeeze())
        G = objective(new_G).item()
        # selection
        if G < objective(torch.tensor(de.fitness[parent_idx], dtype=torch.float)).item():
            de.population[parent_idx] = trial
            de.fitness[parent_idx] = new_G
            if G <= best_observed_obj:
                best_observed_obj = G

        # fit the GP model
        # fit_gpytorch_model(mll)

        # define acquisition function based on fitted GP
        # acqf = qKnowledgeGradient(
        #     model=model,
        #     objective=objective,
        #     num_fantasies=args.acqf_opt_num_fantasies,
        # )

        # optimize acquisition and get new observation via simulation at selected parameters
        # new_theta, new_G, new_G_sem = optimize_acqf_and_get_observation(
        #     acq_func=acqf,
        #     args=args)

        # concatenate observations
        # train_theta = torch.cat([train_theta, new_theta], dim=0)
        # train_G = torch.cat([train_G, new_G], dim=0)
        # train_G_sem = torch.cat([train_G_sem, new_G_sem], dim=0)

        # update progress
        # train_G_objectives = objective(train_G)
        # best_observed_idx = train_G_objectives.argmax()
        # best_observed_obj = train_G_objectives[best_observed_idx].item()
        best_observed.append(best_observed_obj)

        # re-initialize the models so they are ready for fitting on next iteration
        # mll, model = initialize_model(
        #     train_theta,
        #     train_G,
        #     train_G_sem,
        # )

        t1 = time.time()

        # log
        logger.log(
            i=tt,
            time=t1 - t0,
            best=best_observed_obj,
            case_diff=case_diff(new_G),
            objective=objective(new_G).item(),
            theta=unnormalize_theta(new_theta.detach().squeeze())
        )

        # save state
        state = {
            'train_theta' : train_theta,
            'train_G' : train_G,
            'train_G_sem'  : train_G_sem,
            'best_observed_obj': best_observed_obj,
            'best_observed_idx': best_observed_idx
        }
        save_state(state, args.filename)

    # print best parameters
    # print()
    print('FINISHED.')
    # print('Best objective:  ', best_observed_obj)
    # print('Best parameters:')

    # scale back to simulation parameters (from unit cube parameters in BO)
    normalized_calibrated_params = train_theta[best_observed_idx]
    calibrated_params = unnormalize_theta(normalized_calibrated_params)
    pprint.pprint(parr_to_pdict(calibrated_params, measures_optimized=args.measures_optimized))

    with open('logs/{}_betas.pkl'.format(args.filename), 'wb') as f:
        pickle.dump(parr_to_pdict(calibrated_params, measures_optimized=args.measures_optimized), f)
