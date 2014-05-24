/*********************************************************
  hmm2.h
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:19 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int hmm2(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} hmm2REC = {57,3,hmm2};
