import sys

sys.path.append("../")

import torch.nn

from utils.utils import *
import click
import subprocess
from glob import glob
import multiprocessing

device = torch.device('cpu')

envs = ['VizDoom', 'Mario', 'GridWorld', 'Pong']
maps = {
    envs[0]: ['my_way_home_org', 'my_way_home_spwnhard', 'my_way_home_spwnhard_nogoal']
}


def execute_eval(args):
    subprocess.call(args, shell=True)


@click.command()
@click.option('-rd', '--root_dir', type=click.STRING)
@click.option('--start', type=click.INT, default=0)
@click.option('--end', type=click.INT, default=10)
@click.option('--parallel_processes', type=click.INT, default=2)
def evaluate_folder(root_dir, start, end, parallel_processes):
    command_buffer = []

    pool = multiprocessing.Pool(processes=parallel_processes)

    for dirname in glob(os.path.join(root_dir, 'Jul*'))[start:end]:
        command_buffer.append("{} exploration_evaluation.py --dir={}".format(sys.executable, dirname))
    r = pool.map_async(execute_eval, command_buffer)

    r.wait()

    pass


if __name__ == "__main__":
    evaluate_folder()
