"""Command-line wrappers for evaluating saved benchmark samples."""
import argparse
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plus = commands.add_parser('evalplus', help='Delegate to EvalPlus benchmark tests')
    plus.add_argument('--dataset', choices=['humaneval','mbpp'], required=True)
    plus.add_argument('--samples', required=True)
    bcb = commands.add_parser('bcb', help='BigCodeBench remote execution')
    bcb.add_argument('--samples', required=True)
    bcb.add_argument('--endpoint', required=True)
    bcb.add_argument('--subset', choices=['full','hard'], default='full')
    args = parser.parse_args(argv)
    if args.command == 'evalplus':
        subprocess.run([sys.executable, '-m', 'evalplus.evaluate', '--dataset', args.dataset,
                        '--samples', args.samples], check=True)
    else:
        subprocess.run(['bigcodebench.evaluate', 'instruct', args.subset,
                        '--samples', args.samples, '--execution', 'gradio',
                        '--gradio_endpoint', args.endpoint], check=True)
