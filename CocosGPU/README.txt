Do the following steps to set your compiler (if needed):
- edit CC and CFLAGS lines in the Makefile to reflect your 
  compiler and optimiser
To compile:
- run the script “CompileAndRun.sh”

Use:
- ./cocos -h
to print a help message.

In proteins/ there are some examples. 
In particular, you may want to try:
	./cocos -i proteins/1ZDD.fa -a -v
that takes just a FASTA file as input.
Otherwise you can try:
	./cocos -i proteins/1ZDD.in.cocos -v
that uses a complete input file.
