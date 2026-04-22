import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from plots import plot_comparison_trace, plot_diagnostics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mc', help='Path to MC npz')
    parser.add_argument('--pde', help='Path to PDE npz')
    parser.add_argument('--out', help='Output png path')
    parser.add_argument('--type', default='trace', choices=['trace', 'diagnostics'])
    parser.add_argument('--title', default='Comparison')
    args = parser.parse_args()
    
    if args.type == 'trace':
        plot_comparison_trace(args.mc, args.pde, args.out, args.title)
    elif args.type == 'diagnostics':
        plot_diagnostics(args.pde, args.out)

if __name__ == "__main__":
    main()
