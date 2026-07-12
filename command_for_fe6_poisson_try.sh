cd /Users/alexsin/Desktop/SURE/snewpdag
mkdir -p output

python -m snewpdag.trials.Simple Control -n 10 | \
env MPLCONFIGDIR=/private/tmp/snewpdag-mpl \
    TAG=fe6_poisson_try \
    MODEL=s27 \
    SPECIES=ibd \
    CHANNEL=ibd \
    RA=-60.0 \
    DEC=-30.0 \
    PIXEL32=11990 \
    YIELD_SK=7800 \
    YIELD_SNOPLUS=280 \
    YIELD_JUNO=7200 \
    YIELD_LVD=360 \
    DISTANCE=10 \
    LAG_NBINS=200 \
    LAG_WINDOW=1.0 \
    LAG_SCAN_LOW=-0.05 \
    LAG_SCAN_HIGH=0.05 \
    LAG_SCAN_STEP=0.0002 \
    LAG_LEAD_TIME=-0.1 \
    LAG_SIGMA_FUDGE=1.0 \
python -m snewpdag --jsonlines snewpdag/data/fe6_poisson_try.csv
