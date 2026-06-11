# Quantitaive Analysis 

This repository contains quantitative analysis codes for data preprocessing, statistical modeling, 
and time-series analysis, providing flexible tools for making money. Maybe.

This repo is not documented at all. It is for personal use, but feel free to look around.

## Future Plans

**Monotonicity**
Find chart with highest monotonicity on arbitrary but preferably long time intervals.\
Stock, crypto, commodity, currency, ETF, etc.\
See: Quant_Anal\stat\monotonicity.py

**Pattern Matching**  
Cut out windows from charts. These intervals will be *patterns*.
Window width can be as small as two candles.\
Have a distance metric between these patterns.\
Distances can be: L1, L2, correlation, point-pair-wise monotonicity, point-pair-wise change, etc.  
Before distance calculation normalization or standardization can be applied.
Clusterize these patterns based on a distance.\
Find the most distant patterns. These distant patterns can be rare and may suggest special information.

 