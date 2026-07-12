cd /Users/alexsin/Desktop/SURE/snewpdag
mkdir -p output

python -m snewpdag.trials.Simple Control -n 1000 | \
env MPLCONFIGDIR=/private/tmp/snewpdag-mpl \
    TAG=fe5_try \
    MODEL=s27 \
    SPECIES=ibd \
    CHANNEL=ibd \
    RA=-60.0 \
    DEC=-30.0 \
    PIXEL8=749 \
    PIXEL16=2997 \
    YIELD_SK=7800 \
    YIELD_SNOPLUS=280 \
    YIELD_JUNO=7200 \
    YIELD_LVD=360 \
    DISTANCE=10 \
python -m snewpdag --jsonlines --seed 1 snewpdag/data/fe5_try.csv