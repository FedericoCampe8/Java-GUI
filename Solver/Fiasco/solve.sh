BASEDIR=/home/matteo/Desktop/Java-GUI/Solver

$BASEDIR/Fiasco/fiasco \
    --input ../sys/1AIL.in.fiasco \
    --outfile ../proteins/1AIL.out.pdb \
    --domain-size 100 \
    --ensembles 1000000 \
    --timeout-search 60 \
    --timeout-total 120 \
    --distance-leq 42 51 12.442983 \
    --distance-leq 31 60 32.434273 \
    --distance-leq 15 51 16.674183 \
    --distance-leq 12 60 22.955608 \
    --distance-leq 42 12 20.18199 \
    --distance-leq 31 15 10.495963 \
