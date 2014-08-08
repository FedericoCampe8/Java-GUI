BASEDIR=/home/matteo/NetBeansProjects/jafatt/Solver

$BASEDIR/Fiasco/fiasco \
    --input ../proteins/1ZDD.in.fiasco \
    --outfile ../proteins/1ZDD.out.pdb \
    --domain-size 10 \
    --ensembles 1000000 \
    --timeout-search 60 \
    --timeout-total 120 \
    --distance-leq 3 28 1 \
    --distance-leq 11 18 1 \
    --distance-leq 28 3 6.5738773 \
    --distance-leq 18 11 17.434011 \
