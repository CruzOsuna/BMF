#!/usr/bin/env python3

# Script for parsing command line arguments and running single-cell
# data extraction functions
# Joshua Hess

# cli.py
from ParseInput import ParseInput  # Now matches the renamed function
from SingleCellDataExtraction import MultiExtractSingleCells

def main():
    args = ParseInput()  # Now matches the function name
    MultiExtractSingleCells(**args)

if __name__ == '__main__':
    main()