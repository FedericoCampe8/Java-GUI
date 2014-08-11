BASEDIR=/home/matteo/Desktop/Java-GUI/Solver

$BASEDIR/Fiasco/fiasco \
    --input ../proteins/1NIZ.in.fiasco \
    --outfile ../proteins/1NIZ.out.pdb \
    --domain-size 100 \
    --ensembles 1000000 \
    --timeout-search 60 \
    --timeout-total 120 \
    --distance-leq 5 9 8.082712 \
    --distance-leq 2 12 11.797802 \
