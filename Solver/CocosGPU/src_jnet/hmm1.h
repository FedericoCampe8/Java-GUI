/*********************************************************
  hmm1.h
  --------------------------------------------------------
  generated at Wed Jul  7 17:09:18 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int hmm1(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} hmm1REC = {408,3,hmm1};
