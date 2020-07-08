import numpy as np
import pickle, math
import pandas as pd
import multiprocessing

from lib.mobilitysim import MobilitySimulator
from lib.parallel import launch_parallel_simulations
from lib.distributions import CovidDistributions
from lib.data import collect_data_from_df
from lib.measures import (MeasureList, Interval,
                          BetaMultiplierMeasureByType,
                          SocialDistancingForAllMeasure,
                          SocialDistancingForPositiveMeasure,
                          SocialDistancingForPositiveMeasureHousehold)
from lib.inference import gen_initial_seeds, extract_seeds_from_summary
from lib.plot import Plotter
import matplotlib.pyplot as plt


TO_HOURS = 24.0

# Choose random seed
c = 0
# Set it
np.random.seed(c)
# Define prefix string used to save plots
runstr = f'run{c}_'

random_repeats = 40 # Set to at least 40 to obtain stable results

num_workers = multiprocessing.cpu_count() - 1

start_date = '2020-03-08'
end_date = '2020-03-27'
sim_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
max_time = TO_HOURS * sim_days # in hours


from lib.settings.town_settings_tubingen import *
mob_settings = 'lib/mobility/Tubingen_settings_10.pk'
area = 'TU'
country = 'GER'

# See town-generator.ipynb for an example on how to create the settings
with open(mob_settings, 'rb') as fp:
    obj = pickle.load(fp)
mob = MobilitySimulator(**obj)


new_cases_ = collect_data_from_df(country=country, area=area, datatype='new',
    start_date_string=start_date, end_date_string=end_date)
new_cases = np.ceil(
        (new_cases_ * mob.num_people_unscaled) /
        (mob.downsample * mob.region_population))

plt.plot(new_cases.sum(1));

distributions = CovidDistributions(country=country)

def save_summary(summary, filename):
    with open('summaries/' + filename, 'wb') as fp:
        pickle.dump(summary, fp)

def load_summary(filename):
    with open('summaries/' + filename, 'rb') as fp:
        summary = pickle.load(fp)
    return summary

heuristic_seeds = True

# (a) define heuristically based on true cases and literature distribution estimates
if heuristic_seeds:
    initial_seeds = gen_initial_seeds(new_cases)

# (b) define based state of previous batch of simulations,
# using the random rollout that best matched the true cases in terms of squared error
else:
    seed_summary_ = load_summary('summary_example.pk')
    seed_day_ = 7
    initial_seeeds = extract_seeds_from_summary(seed_summary_, seed_day_, new_cases)

import sys
with open('logs/{}_betas.pkl'.format(sys.argv[1]), 'rb') as f:
    inferred_params = pickle.load(f)

print("Loading betas: \n")
print(inferred_params)

# inferred_params = {
#     'beta_household': 1.3125,
#     'betas': {
#         'bus_stop': 0.9375,
#         'education': 0.5625,
#         'office': 0.1875,
#         'social': 0.5625,
#         'supermarket': 1.3125
#     }
# }

def standard_testing(max_time):
    daily_increase = new_cases.sum(axis=1)[1:] - new_cases.sum(axis=1)[:-1]
    standard_testing_params = {
        'testing_t_window'    : [0.0, max_time], # in hours
        'testing_frequency'   : 1 * TO_HOURS,     # in hours
        'test_reporting_lag'  : 2 * TO_HOURS,     # in hours (actual and self-report delay)
        'tests_per_batch'     : int(daily_increase.max()), # test capacity based on empirical positive tests
        'test_fpr'            : 0.0, # test false positive rate
        'test_fnr'            : 0.0, # test false negative rate
        'test_smart_delta'    : 3 * TO_HOURS, # in hours
        'test_smart_duration' : 7 * TO_HOURS, # in hours
        'test_smart_action'   : 'isolate',
        'test_smart_num_contacts'   : 10,
        'test_targets'        : 'isym',
        'test_queue_policy'   : 'fifo',
        'smart_tracing'       : None,
    }
    return standard_testing_params

def run(tparam, measure_list, t, local_seeds, dynamic_tracing=False):

    # add standard measure of positives staying isolated
    measure_list +=  [
        SocialDistancingForPositiveMeasure(
            t_window=Interval(0.0, t), p_stay_home=1.0),

        SocialDistancingForPositiveMeasureHousehold(
            t_window=Interval(0.0, t), p_isolate=1.0)
    ]
    measure_list = MeasureList(measure_list)

    # run simulations
    summary = launch_parallel_simulations(
        mob_settings,
        distributions,
        random_repeats, num_workers,
        inferred_params, local_seeds, tparam, measure_list,
        max_time=t,
        num_people=mob.num_people,
        num_sites=mob.num_sites,
        site_loc=mob.site_loc,
        home_loc=mob.home_loc,
        dynamic_tracing=dynamic_tracing,
        verbose=False)
    return summary


lockdown_at_day = 12

example_measures = [

    # education, social sites, and offices close after 1 week
    BetaMultiplierMeasureByType(
        t_window=Interval(lockdown_at_day * TO_HOURS, max_time),
        beta_multiplier={
            'education': 0.0,
            'social': 0.0,
            'bus_stop': 1.0,
            'office': 0.0,
            'supermarket': 1.0
        }),

    # less activities of all due to contact constraints after 1 week
    SocialDistancingForAllMeasure(
     t_window=Interval(lockdown_at_day * TO_HOURS, max_time),
        p_stay_home=0.5)
]

testing_params = standard_testing(max_time)
summary_example = run(testing_params, example_measures, max_time, initial_seeds)
save_summary(summary_example, 'summary_example.pk')

summary_example = load_summary('summary_example.pk')

plotter = Plotter()
plotter.plot_positives_vs_target(
    summary_example, new_cases.sum(axis=1),
    title='Example',
    filename=runstr + 'ex_00',
    figsize=(6, 4),
    start_date=start_date,
    errorevery=1, acc=1000,
    lockdown_at=lockdown_at_day,
    ymax=50)

plotter = Plotter()
plotter.plot_daily_rts(
    summary_example,
    filename=runstr + 'ex_01',
    start_date=start_date,
    sigma=None,
    figsize=(6, 4),
    subplot_adjust=None,
    lockdown_label='Lockdown',
    lockdown_at=lockdown_at_day,
    lockdown_label_y=2.6,
    ymax=4.0)


targets = new_cases.sum(axis=1)
targets = np.hstack((targets[0], np.diff(targets)))

plotter = Plotter()
plotter.plot_daily_infected(
    summary_example,
    title='',
    filename=runstr + 'ex_02',
    errorevery=100, acc=1000,
    figsize=(6, 4),
    start_date=start_date,
    show_target=targets,
    lockdown_at=lockdown_at_day,
    lockdown_label_y=30,
    lockdown_label='Lockdown',
    ymax=50)

plotter = Plotter()
plotter.plot_cumulative_infected(
    summary_example,
    title='',
    filename=runstr + 'ex_03',
    figsize=(6, 4),
    legend_loc='upper left',
    errorevery=100, acc=1000,
    start_date=start_date,
    lockdown_at=lockdown_at_day,
    lockdown_label_y=60,
    show_target=new_cases.sum(axis=1),
    ymax=60)
