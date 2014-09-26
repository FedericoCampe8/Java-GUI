BASEDIR=/Users/nando/Proteins/Java-GUI-master/Solver

$BASEDIR/Fiasco/fiasco \
    --input ../sys/1ZDD.in.fiasco \
    --outfile ../proteins/1ZDD.out.pdb \
    --domain-size 100 \
    --ensembles 1000 \
    --timeout-search 60 \
    --timeout-total 120 \
