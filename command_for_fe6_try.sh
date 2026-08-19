# model can be s27 or s11

cd /Users/alexsin/Desktop/SURE/snewpdag
mkdir -p output

python -m snewpdag.trials.Simple Control -n 10 | \
env MPLCONFIGDIR=/private/tmp/snewpdag-mpl \
    TAG=fe6_try \
    MODEL=s27 \
    SPECIES=ibd \
    CHANNEL=ibd \
    RA=-60.0 \
    DEC=-30.0 \
    PIXEL8=749 \
    PIXEL32=11990 \
    YIELD_SK=7800 \
    YIELD_SNOPLUS=280 \
    YIELD_JUNO=7200 \
    YIELD_LVD=360 \
    DISTANCE=10 \
python -m snewpdag --jsonlines snewpdag/data/fe6_try.csv