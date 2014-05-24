install: jnet

clean: 
	rm *.o; rm jnet

OBJECTS	= jnet.o net1.o net1b.o net2.o net2b.o psinet1.o psinet1b.o psinet1c.o \
	psinet2.o  psinet2b.o psinet2c.o hmm1.o hmm2.o consnet.o psisol25.o hmmsol25.o \
	psisol0.o psisol5.o hmmsol0.o hmmsol5.o

CC	= gcc
CFLAGS	= -O2
LFLAGS	= -lm

jnet:  $(OBJECTS) 
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) $(LFLAGS)

net1.o: net1.c
	$(CC) $(CFLAGS) -c net1.c
net1b.o: net1b.c
	$(CC) $(CFLAGS) -c net1b.c
net2.o: net2.c
	$(CC) $(CFLAGS) -c net2.c
net2b.o: net2b.c
	$(CC) $(CFLAGS) -c net2b.c
psinet1.o: psinet1.c
	$(CC) $(CFLAGS) -c psinet1.c
psinet1b.o: psinet1b.c
	$(CC) $(CFLAGS) -c psinet1b.c
psinet1c.o: psinet1c.c
	$(CC) $(CFLAGS) -c psinet1c.c
psinet2.o: psinet2.c
	$(CC) $(CFLAGS) -c psinet2.c
psinet2b.o: psinet2b.c
	$(CC) $(CFLAGS) -c psinet2b.c
psinet2c.o: psinet2c.c
	$(CC) $(CFLAGS) -c psinet2c.c
hmm1.o: hmm1.c
	$(CC) $(CFLAGS) -c hmm1.c
hmm2.o: hmm2.c
	$(CC) $(CFLAGS) -c hmm2.c
consnet.o: consnet.c
	$(CC) $(CFLAGS) -c consnet.c
hmmsol25.o: hmmsol25.c
	$(CC) $(CFLAGS) -c hmmsol25.c
hmmsol5.o: hmmsol5.c
	$(CC) $(CFLAGS) -c hmmsol5.c
hmmsol0.o: hmmsol0.c
	$(CC) $(CFLAGS) -c hmmsol0.c
psisol25.o: psisol25.c
	$(CC) $(CFLAGS) -c psisol25.c
psisol0.o: psisol0.c
	$(CC) $(CFLAGS) -c psisol0.c
psisol5.o: psisol5.c
	$(CC) $(CFLAGS) -c psisol5.c
