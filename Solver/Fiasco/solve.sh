BASEDIR=/home/matteo/Desktop/Java-GUI/Solver

$BASEDIR/Fiasco/fiasco \
    --input ../sys/1ZDD.in.fiasco \
    --outfile ../proteins/1ZDD.out.pdb \
    --domain-size 100 \
    --ensembles 1000000 \
    --timeout-search 60 \
    --timeout-total 120 \
    --distance-leq 28 3 9.040053 \
    --distance-geq 28 3 1.8080107 \
    --distance-leq 18 12 6.535852 \
    --distance-geq 18 12 1.3071704 \
    --distance-leq 23 7 10.499124 \
    --distance-geq 23 7 2.0998247 \
